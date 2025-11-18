"""
Tucker2-ALS-Sigma Algorithm Implementation on a given Torch.Tensor and the ranks
"""

import torch
import cupy as cp
from cupyx.scipy.sparse.linalg import minres
import argparse

#Reconstructing a tensor from a Tucker decomposition (ie. the core and the factors)
def reconstruct_tensor_from_factors(core, factors):
    return torch.tensordot(torch.tensordot( factors[0], core, dims = [[1], [0]]) , factors[1], dims = [[3], [1]])

#Return the approximation error of tensor by tensor2 in respect to the norm sigma
def calcul_err_sigma(tensor, sigma, tensor2):
    X_diff = tensor - tensor2
    return torch.norm(torch.matmul(sigma , X_diff.reshape((-1, tensor.size()[3]))))

def matvec_M(x, A, D, r, b):
    """ Compute (A^t  A) * (D^t D) @ x to faster the algorithm for large size matrices"""
    A = cp.asarray(A)
    D = cp.asarray(D)
    b = cp.asarray(b)
    r, n2, n3, s, n4, n5 = D.shape
    n1, n2, n3,n7, n8, n9 = A.shape
    x = x.reshape(n1,r)
    x = torch.utils.dlpack.from_dlpack(x.toDlpack())
    x = cp.asarray(x)
    result = cp.einsum('ijkumn,qjksmn,us -> iq', A, D, x)
    return result.ravel()

def solve_M_B(A, D, b, r, n):
    """ Solve (MU = b) for U using the MINRES method"""
    A = cp.asarray(A)
    D = cp.asarray(D)
    b = cp.asarray(b)
    M_operator = cp.sparse.linalg.LinearOperator((n, n), matvec=lambda v: matvec_M(v, A, D, r,b), dtype=cp.float32)
    x, istop = minres(M_operator, b,tol=1e-10,maxiter=2000)[:2] 
    return x.astype(cp.float32)

def partial_tucker_sigma(
    tensorT: torch.Tensor,
    rank,
    sigma: torch.Tensor,
    init='svd',
    tinit= None ,
    n_iter_max= int,
    tol=1e-6,
    verbose=1,
    cvg_criterion="abs_rec_error",
):
    """
    For an easier comparison with the classical Tucker algorithm, we used the same strcuture as the Tucker2 function of Tensorly
    Tucker-2 decomposition via Alternating Least Squares (ALS) with respect to a Sigma norm.

    Computes a rank-`rank` Tucker decomposition of the input `tensorT` such that:

        tensorT ≈ core ×₁ factors[0] ×₂ factors[1] 

    This version of ALS incorporates a Sigma-weighted norm during optimization, allowing
    for distribution-aware compression. The algorithm seeks to minimize the reconstruction
    error under the Sigma norm.

    Parameters
    ----------
    tensorT : torch.Tensor
        Input tensor to decompose.

    rank : int or list[int]
        Target multilinear rank of the decomposition.

    sigma : torch.Tensor
        Sigma weighting tensor used for the Sigma-norm during optimization.

    init : {'svd', 'random'}, optional (default: 'svd')
        Initialization method for the factor matrices.

    tinit : torch.Tensor or None
        Optional initial core tensor.

    n_iter_max : int
        Maximum number of ALS iterations.

    tol : float
        Tolerance for convergence.

    verbose : int
        Verbosity level (0: silent, 1: progress messages).

    cvg_criterion : {'abs_rec_error', 'rel_rec_error'}, optional (default: 'abs_rec_error')
        Criterion to determine convergence.

    Returns
    -------
    core : torch.Tensor
        Core tensor of the Tucker decomposition.

    factors : list[torch.Tensor]
        List of factor matrices corresponding to each mode.
    """
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Take transpose of Sigma for simplicity
    sigma = torch.t(sigma)
    sigma= sigma.to(device)

    # Move the first axis to last axis and Turn the tensor to be 4-th order
    tensor = torch.moveaxis(tensorT, 0, -1)
    tensor = tensor[:, :, None, :]
    tensor = tensor.to(device)

    rank=[rank[1], rank[0]]
    tsize=tensor.size()
    norm_sigma_tensor=torch.norm(torch.flatten(torch.matmul(sigma, tensor.reshape((-1, tsize[3])))))


    #Initialization of the weights and factors
    if tinit != None:
        core,factors = tinit
        core = core[:]
        factors = factors[:]
        Q,R = torch.linalg.qr(factors[0])
        factors[0] = Q
        core = torch.tensordot(R, core, dims=[[1],[0]])
        Q,R = torch.linalg.qr(factors[1])
        factors[1] = Q
        core = torch.tensordot(core, R, dims=[[-1],[1]])
    else:
        if init == 'random':
            factors = [torch.rand([tsize[0],rank[0]]).to(device) , torch.rand([tsize[3],rank[1]]).to(device)]
            core = torch.tensordot(torch.tensordot( factors[0], tensor, dims = [[0], [0]]) , factors[1], dims = [[3], [0]]).to(device)
        
        if init == 'svd':
            U, _, _ = torch.linalg.svd(tensor.reshape((tsize[0],-1)) ,full_matrices = False)
            factors = [U[:, :rank[0]].to(device)]
            
            U, _, _ = torch.linalg.svd(torch.t(tensor.reshape((-1,tsize[3]))) ,full_matrices = False)                
            factors += [U[:, :rank[1]]]

            #Reshape Sigma and multiply with first factor matrix
            sigma2 = sigma.reshape((-1,)+tuple(tsize[i] for i in range(3)))
            sigma2 = torch.tensordot(sigma2, factors[0], dims=[[1], [0]]).moveaxis(-1,1).reshape((sigma2.size()[0], -1))
            
            tensor2 = tensor.reshape((-1,tsize[3]))
            sigmatensor = sigma @ tensor2
            DX = torch.linalg.lstsq(torch.t(sigma2) @ sigma2, torch.t(sigma2)@ sigmatensor)[0]
            core = torch.t( torch.linalg.solve(torch.t(factors[1]) @ factors[1], torch.t(factors[1]) @ torch.t(DX))).reshape([rank[0], tsize[1], tsize[2], rank[1]]).to(device)

    core_factors_checkpoint = [core, factors]

    rec_errors = []

    unnorml_rec_error = calcul_err_sigma(
        tensor, sigma, reconstruct_tensor_from_factors(core, factors)
    )
    
    rec_error = unnorml_rec_error / norm_sigma_tensor
    rec_errors.append(rec_error)

    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in [0,3,4]:
            if verbose > 1:
                print("Mode", mode, "of", 3)
                
            if mode == 0:
                # Compute the tensor contraction over the first dimension and reshape it
                sigmaTsigma = torch.tensordot(sigma, sigma , dims=[ [0] , [0] ])
                sigmaTsigma_reshaped = sigmaTsigma.reshape(list(tsize[:3]) + list(tsize[:3]))

                # Contract the last dimension of 'core' with the second dimension of 'factors[1]'
                Core_factor = torch.tensordot(core, factors[1], dims=[[-1], [1]]).to(device)
                
                # Compute a contraction over the last dimension
                Corefactor_2 = torch.tensordot(Core_factor, Core_factor, dims=[ [-1] , [-1] ])
                
                tensor_reshaped = tensor.reshape((-1,tsize[3])) @ factors[1]
                sigmatensor = sigma @ tensor_reshaped
                sigma2tensor = torch.tensordot(sigma, sigmatensor, dims=[[0], [0]]).reshape(list(tsize)[:3]+[-1])
                sigmatensorcore = torch.tensordot(sigma2tensor, core, dims=[[1,2,3], [1,2,3]]).flatten() 

                # Solve first factor matrix 
                
                factor = solve_M_B(sigmaTsigma_reshaped,Corefactor_2,sigmatensorcore, rank[0], sigmatensorcore.size(0)).reshape(tsize[mode],rank[0])
                factor = torch.utils.dlpack.from_dlpack(factor.toDlpack())
                factors[0] = factor.to(device)
                
            elif mode == 3:
                # Compute the tensor contraction: Sigma @ SecondFactorMatrix @ Core
                sigma_reshaped = sigma.reshape((-1,)+tuple(tsize[i] for i in range(3)))
                sigma_factor = torch.tensordot(sigma_reshaped, factors[0], dims=[[1], [0]]).moveaxis(-1,1)
                sigma_factorcore = torch.tensordot(sigma_factor, core, dims=[[1,2,3], [0,1,2]])
                
                # Solve second factor matrix 
                b = torch.t(sigma_factorcore) @ sigma @ tensor.reshape((-1,tsize[3]))
                factors[1] = torch.t(torch.linalg.solve(torch.t(sigma_factorcore) @ sigma_factorcore, b)).reshape((-1,rank[1]))
                
            else :
                # Reshape Sigma matrix and multiply it with first factor matrix
                sigma_reshaped = sigma.reshape((-1,)+tuple(tsize[i] for i in range(3)))
                sigma_factor = torch.tensordot(sigma_reshaped, factors[0], dims=[[1], [0]]).moveaxis(-1,1).reshape((sigma2.size()[0], -1))
                
                # Solve core tensor 
                tensor_reshaped = tensor.reshape((-1,tsize[3])) 
                sigmatensor = sigma @ tensor_reshaped
                DX = torch.linalg.solve(torch.t(sigma_factor) @ sigma_factor, torch.t(sigma_factor)@ sigmatensor)
                core = torch.t( torch.linalg.solve(torch.t(factors[1]) @ factors[1], torch.t(factors[1]) @ torch.t(DX))).reshape(core.size())
                
        if tol:
            unnorml_rec_error = calcul_err_sigma(
                tensor, sigma, reconstruct_tensor_from_factors(core, factors)
            )
            
            rec_error = unnorml_rec_error / norm_sigma_tensor
            rec_errors.append(rec_error)

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print(
                        f"iteration {iteration}, reconstruction error: {rec_error}, decrease = {rec_error_decrease}, unnormalized = {unnorml_rec_error}"
                    )
                
                if rec_error_decrease < 0 :
                    core, factors = core_factors_checkpoint[:]
                    print('divergence, stopped before')
                    break

                if cvg_criterion == "abs_rec_error":
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print(f"Tucker converged after {iteration} iterations")
                    break

            else:
                if verbose:
                    print(f"reconstruction error={rec_errors[-1]}")
        core_factors_checkpoint = [core[:], factors[:]]

    return (core, [factors[1], factors[0]])
def main():
    parser = argparse.ArgumentParser(description="Run Tucker2-ALS-Sigma Algorithm.")
    
    parser.add_argument('--tensor_path', type=str, required=True,
                        help='Path to the input tensor (.pt file)')
    parser.add_argument('--rank', type=int, nargs='+', required=True,
                        help='Target rank(s) for decomposition')
    parser.add_argument('--sigma_path', type=str, required=True,
                        help='Path to the sigma tensor (.pt file)')
    parser.add_argument('--init', type=str, default='svd', choices=['svd', 'random'],
                        help='Initialization method')
    parser.add_argument('--tinit', type=str, default=None,
                        help='Optional tensor initialization (.pt file)')
    parser.add_argument('--n_iter_max', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Tolerance for convergence')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    parser.add_argument('--cvg_criterion', type=str, default='abs_rec_error',
                        help='Convergence criterion')

    args = parser.parse_args()

    tensorT = torch.load(args.tensor_path)
    sigma = torch.load(args.sigma_path)
    tinit = torch.load(args.tinit) if args.tinit else None

    partial_tucker_sigma(tensorT=tensorT,
                      rank=args.rank,
                      sigma=sigma,
                      init=args.init,
                      tinit=tinit,
                      n_iter_max=args.n_iter_max,
                      tol=args.tol,
                      verbose=args.verbose,
                      cvg_criterion=args.cvg_criterion)

if __name__ == "__main__":
    main()

