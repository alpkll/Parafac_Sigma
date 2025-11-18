
"""
Covariance computation

Tools to compute the covariance of the input of each convolutional layer of a neural network.
"""
import argparse
from time import perf_counter
from psutil import Process as psutil_Process
import torch
from torch import nn
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)
from tqdm import tqdm
from datetime import datetime
import os 
from torchinfo import summary
import nvidia_smi
import re
from functools import reduce
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

def get_module_by_name(model: nn.Module, name: str):
    names = name.split('.')
    return reduce(getattr, names, model)
def conv2d_name_list(model: nn.Module) -> list[str]:
    """
    Return the list of the names of the conv2d layers of a model.

    Parameters
    ----------
    model : nn.Module
        the original PyTorch model

    Returns
    -------
    list[str]
        the list of the names of the conv2d layers of the model
    """
    conv2d_layers = []
    evaluation_nodes = get_graph_node_names(model)[1]

    for key in dict(model.named_parameters()).keys():

        # if the parameter is a bias, we already compressed the corresponding layer with the parameter weights
        if key.endswith('bias'):
            continue

        # remove '.weight' or '.bias' strings from the key using a regex
        layer = re.sub(r'\.\w+$', '', key)

        # get the module by name
        module = get_module_by_name(model, layer)

        # check if the corresponding module is a conv2d layer
        if isinstance(module, torch.nn.Conv2d) and (module.kernel_size !=
                                                    (1, 1)):
            if layer in evaluation_nodes:
                conv2d_layers.append(layer)

    return conv2d_layers
def gpu_memory_usage(verbose=True) -> tuple[int, int, int]:
    """
    Return (and print if verbose) the GPU memory usage.
    - GPU memory
    - GPU memory allocated
    - GPU memory free

    A GPU is required.
    """
    if torch.cuda.is_available():
        nvidia_smi.nvmlInit()
        index = torch.cuda.current_device()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = info.total
        gpu_memory_allocated = info.used
        gpu_memory_free = info.free
        if verbose:
            print(f'GPU memory: {gpu_memory / 10 ** 9:.2e} GB VRAM',
                  f"GPU memory allocated: {gpu_memory_allocated / 1e9:.2e} GB VRAM",
                  f"GPU memory free: {gpu_memory_free / 1e9:.2e} GB VRAM")
        return gpu_memory, gpu_memory_allocated, gpu_memory_free
    else:
        print('No GPU available')
        return 0, 0, 0


def get_kernel_input_sizes(model: nn.Module,
                           input_size=(1, 3, 224, 224)
                           ) -> tuple[dict[str, dict[str, int | tuple]], dict[str, int]]:
    """For each conv2d of the model, get the kernel size and input size.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.
    input_size : tuple, optional
        The input size of the model, by default (1, 3, 224, 224)
    
    Returns
    -------
    dict[str, dict[str, int | tuple]]
        A dictionary with the name of the conv2d as key and a dictionary
        with:
            - kernel_input_size: the kernel input size of the conv2d (s, hd, wd)
            - input_size: the input size of the conv2d (s, h, w)
            - kernel_input_reals: the number of reals in the input of the kernel (s x hd x wd)
            - input_reals: the number of reals in the input of the conv2d (s x h x w)
            - intermediate_size: the size of the intermediate matrix (s, h, w, s, hd, wd)
            - intermediate_reals: the number of reals in the intermediate matrix (s x h x w x s x hd x wd)
            - final_size: the size of the final covariance matrix (s, hd, wd, s, hd, wd)
            - final_reals: the number of reals in the final covariance matrix (s x hd x wd x s x hd x wd)
    """
    # Get the conv2d names
    conv2d_names = conv2d_name_list(model)
    # Get the summary of the model
    summary_model = summary(model, input_size=input_size, verbose=0)

    info = {}
    total = {'intermediate_reals': 0, 'final_reals': 0}
    i = 0
    for module in summary_model.summary_list:
        if isinstance(module.module, nn.Conv2d) and module.module.kernel_size != (1, 1):
            conv_name = conv2d_names[i]
            i += 1
            info[conv_name] = {
                'kernel_input_size': (module.module.in_channels,) + module.module.kernel_size,
                'input_size': tuple(module.input_size),
            }
            info[conv_name]['kernel_input_reals'] = int(np.prod(info[conv_name]['kernel_input_size']))
            info[conv_name]['input_reals'] = int(np.prod(info[conv_name]['input_size']))
            info[conv_name]['intermediate_size'] = (info[conv_name]['input_size'][:1] 
                                                    + info[conv_name]['kernel_input_size'][:-1] 
                                                    + (2 * info[conv_name]['kernel_input_size'][2] + 1,) 
                                                    + info[conv_name]['input_size'][1:])
            info[conv_name]['intermediate_reals'] = int(np.prod(info[conv_name]['intermediate_size']))
            info[conv_name]['final_size'] = info[conv_name]['kernel_input_size'] + info[conv_name]['kernel_input_size']
            info[conv_name]['final_reals'] = int(np.prod(info[conv_name]['final_size']))

            total['intermediate_reals'] += info[conv_name]['intermediate_reals']
            total['final_reals'] += info[conv_name]['final_reals']
    return info, total


def show_kernel_input_sizes(model: nn.Module,
                            input_size=(1, 3, 224, 224),
                            float_size=32
                            ) -> None:
    """For each conv2d of the model, show the intermediate and final size.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.
    input_size : tuple, optional
        The input size of the model, by default (1, 3, 224, 224)
    
    Returns
    -------
    None
    """
    assert float_size % 8 == 0, 'The float size must be a multiple of 8.'
    float_size = float_size // 8

    info, total = get_kernel_input_sizes(model, input_size=input_size)
    for conv_name, conv_info in info.items():
        print(f'{conv_name}:')
        print(f"\tintermediate size: {conv_info['intermediate_size']}")
        print(f"\tintermediate weight: {float_size * conv_info['intermediate_reals'] / 1e9:.3f} GB")
        print(f"\tfinal size: {conv_info['final_size']}")
        print(f"\tfinal weight: {float_size * conv_info['final_reals'] / 1e9:.3f} GB")

    print(f'Total:')
    print(f"\tintermediate weight: {float_size * total['intermediate_reals'] / 1e9:.3f} GB")
    print(f"\tfinal weight: {float_size * total['final_reals'] / 1e9:.3f} GB")

def pre_convolution_values_extractor(model: nn.Module) -> tuple[GraphModule, set[str]]:
    """
    Create a feature extractor that returns the input of each conv2d layer.

    Use:
    outputs = pre_convolution_values_extractor(model)
    for batch in dataloader:
        output = outputs(batch[0])
        for layer in conv2d_layers:
            layer_input[layer] = output[layer]

    WARNING:
    if the computation graph of the model is not the same as the one used to train the model,
    like in inception models, the function will not work.
    """
    # identify the name of the layer that is the input of each conv2d layer
    _, possible_return_nodes = get_graph_node_names(model)
    conv2d_layers: set[str] = set()
    
    return_nodes = dict()
    for layer_name, layer in filter(lambda x: not list(x[1].children()), model.named_modules()):
        if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size != (1, 1):
            if layer_name in possible_return_nodes:
                i = possible_return_nodes.index(layer_name)
                return_nodes[possible_return_nodes[i - 1]] = layer_name
                conv2d_layers.add(layer_name)
    print(return_nodes)

    # create the feature extractor
    return create_feature_extractor(model, return_nodes=return_nodes), conv2d_layers


def get_mean_pre_conv_input_full(model: nn.Module,
                                 dataloader: DataLoader
                                 ) -> dict[str, torch.Tensor]:
    """
    Get the mean of the input of all the conv2d layers.
    Does not change the size of the input to correspond to the size of the kernel.

    Parameters
    ----------
    model : nn.Module
        the original PyTorch model
    dataloader : DataLoader
        the dataloader of the dataset

    Returns
    -------
    dict[str, torch.Tensor] : dict[str, S x H x W]
        the mean of the input of all the conv2d layers

    """
    print("Start get_mean_pre_conv_input_full")
    model.eval()

    outputs, conv2d_layers = pre_convolution_values_extractor(model)
    print(conv2d_layers)

    mean_pre_conv_input = {}

    # initialize the mean of the input of each conv2d layer
    for layer in conv2d_layers:
        mean_pre_conv_input[layer] = 0

    device = next(model.parameters()).device

    # iterate over the dataset
    for input_images, _ in tqdm(dataloader):
        input_images = input_images.to(device)
        output = outputs(input_images)
        for layer in conv2d_layers:
            mean_pre_conv_input[layer] += output[layer].detach()
    print(f"Full mean RAM use: {psutil_Process().memory_info().rss / 1e9} GB RAM")
    gpu_memory_usage()
    print(f"Full mean are in {output[layer].device}")

    # compute the mean
    for layer in conv2d_layers:
        mean_pre_conv_input[layer] = torch.sum(
            mean_pre_conv_input[layer], axis=0) / len(dataloader.dataset)

    return mean_pre_conv_input


def compute_kernel_mean_reshape(full_mean: torch.Tensor,
                                conv: nn.Conv2d) -> torch.Tensor:
    """
    Compute the mean of the input of a conv2d layer in the shape of the kernel.
    Takes the mean of the input of the conv2d layer not reshaped as input.
    S x H x W -> S x Hd x Wd

    Parameters
    ----------
    full_mean : torch.Tensor
        the mean of the input of the conv2d layer not reshaped
    conv : nn.Conv2d
        the conv2d layer

    Returns
    -------
    torch.Tensor: S x Hd x Wd
        the mean of the input of the conv2d layer in the shape of the kernel
    """

    # WARNING: should we use the same padding as in the conv2d layer?
    padded_full_mean = nn.functional.pad(
        full_mean,
        (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0]),
        mode='constant',
        value=0)
    kernel_mean = torch.zeros((conv.in_channels, ) + conv.kernel_size, device=full_mean.device)
    for i in range(conv.kernel_size[0]):
        for j in range(conv.kernel_size[1]):
            kernel_mean[:, i, j] = torch.mean(
                padded_full_mean[:,
                                 i: padded_full_mean.shape[1] + i - conv.kernel_size[0] + 1: conv.stride[0],
                                 j: padded_full_mean.shape[2] + j - conv.kernel_size[1] + 1: conv.stride[1]],
                dim=(1, 2))
    return kernel_mean


def compute_outer_product(tensor: torch.Tensor
                          ) -> torch.Tensor:
    """
    Return a tensor T such that T[i, j] = kernel_mean[i] * kernel_mean[j]
    with i, j tuples of indices

    If kernel_mean is of shape (N, M), the output is of shape (N, M, N, M)

    S x Hd x Wd -> S x Hd x Wd x S x Hd x Wd

    Parameters
    ----------
    tensor : torch.Tensor
        a tensor of shape S x Hd x Wd

    Returns
    -------
    torch.Tensor: S x Hd x Wd x S x Hd x Wd
        the outer product of the tensor
    """

    return tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * tensor.unsqueeze(
        0).unsqueeze(0).unsqueeze(0)
    # tensor_shape = tensor.shape
    # tensor = tensor.flatten()
    # return torch.einsum('i,j->ij', tensor, tensor).reshape(tensor_shape + tensor_shape)


def get_product_mean_pre_conv_input(model: nn.Module,
                                    dataloader: DataLoader
                                    ) -> dict[str, torch.Tensor]:
    """
    Get the product of the mean of the input of all the conv2d layers in the shape of the kernel.
    E(X_i)E(X_j) for all i, j in the kernel.

    Parameters
    ----------
    model : nn.Module
        the original PyTorch model
    dataloader : DataLoader
        the dataloader of the dataset

    Returns
    -------
    dict[str, torch.Tensor]: dict[str, K x K]
        the mean of the input of all the conv2d layers
    """
    print("Computing the mean of the input of all the conv2d layers...")
    full_means = get_mean_pre_conv_input_full(model, dataloader)
    print("Computing the product of the mean of the input of all the conv2d layers...")
    kernel_product_means = dict()
    for layer in full_means:
        print(f"Computing the product of the mean of the input of {layer}...")
        layer_conv = get_module_by_name(model, layer)
        kernel_product_means[layer] = compute_outer_product(
            compute_kernel_mean_reshape(full_means[layer], layer_conv))

    print(f"Product_mean is in {kernel_product_means[layer].device}")
    return kernel_product_means


@torch.jit.script
def compute_product(input_tensor: torch.Tensor,
                    hd: int,
                    wd: int,
                    output_tensor: torch.Tensor
                    ) -> None:
    """
    Compute the product of the input of a conv2d layer in the shape of the input.
    Return T  of size B x (S x Hd x (2Wd + 1)) x I such that:
    Hd > h >= 0
    Wd > w > -Wd
    T[b, s, h, w, x, y, z] = I[b, x, y, z] * I[b, s, y + h, z + w]

    In practice:
    Hd > h >= 0
    2Wd + 1 > w >= 0


    I -> (S x Hd x (2Wd + 1)) x I

    Parameters
    ----------
    input_tensor : torch.Tensor
        I : B x S x H x W the input of the conv2d layer not reshaped
    hd : int
        the height of the kernel
    wd : int
        the width of the kernel
    output_tensor : torch.Tensor, optional
        (S x Hd x (2Wd + 1)) x I
        add the product of the input of the conv2d layer in the shape of the input to this tensor
    """
    _, _, H, W = input_tensor.shape
    # fp = torch.zeros((B, S, Hd, Wd, S, H, W))
    fp = output_tensor

    for h in range(hd):
        # w = 0
        fp[:, :, h, wd, :, :H - h, : W
           ] += input_tensor[:, :, :H - h, :].unsqueeze(1) * input_tensor[:, :, h:, :].unsqueeze(2)
        for w in range(1, wd):
            #  w > 0
            fp[:, :, h, w + wd, :, :H - h, :W - w
               ] += input_tensor[:, :, :H - h, :W - w].unsqueeze(1) * input_tensor[:, :, h:, w:].unsqueeze(2)
            # w < 0
            fp[:, :, h, wd - w, :, :H - h, w:
               ] += input_tensor[:, :, :H - h, w:].unsqueeze(1) * input_tensor[:, :, h:, :W-w].unsqueeze(2)


def get_product_pre_conv_input_full(model: nn.Module,
                                    dataloader: DataLoader
                                    ) -> dict[str, torch.Tensor]:
    """
    Get the product of the input of all the conv2d layers.
    Does not change the size of the input to correspond to the size of the kernel.

    Parameters
    ----------
    model : nn.Module
        the original PyTorch model
    dataloader : DataLoader
        the dataloader of the dataset

    Returns
    -------
    dict[str, torch.Tensor]: dict[str, (S x H x (2W + 1)) x I]
        the product of the input of all the conv2d layers
    """

    model.eval()

    outputs, conv2d_layers_names = pre_convolution_values_extractor(model)

    conv2d_layers_kernel_size = {
        layer: get_module_by_name(model, layer).kernel_size
        for layer in conv2d_layers_names
    }

    mean_product_pre_conv = {}
   
    # initialize the mean of the input of each conv2d layer
    input_size = tuple(next(iter(dataloader))[0].shape)
    print('input_size',input_size)
    model_infos = get_kernel_input_sizes(model, input_size)[0]
    print("Model infos", model_infos.keys())
    for layer in conv2d_layers_names:
        intermediate_size = model_infos[layer]["intermediate_size"]
        print(intermediate_size)
        mean_product_pre_conv[layer] = torch.zeros(
            intermediate_size, device=next(model.parameters()).device)

    device = next(model.parameters()).device

    # iterate over the dataset
    for input_images, _ in tqdm(dataloader):
        input_images = input_images.to(device)
        output = outputs(input_images)
        for layer_name, conv_kernel_size in conv2d_layers_kernel_size.items():
            compute_product(
                output[layer_name].detach(), 
                conv_kernel_size[0],
                conv_kernel_size[1], 
                mean_product_pre_conv[layer_name])
    print(f"Full product RAM use: {psutil_Process().memory_info().rss / 1e9} GB RAM")
    gpu_memory_usage()
    print(f"Full product is in {mean_product_pre_conv[layer_name].device}")

    # compute the mean
    for layer in conv2d_layers_names:
        mean_product_pre_conv[layer] = torch.sum(
            mean_product_pre_conv[layer], axis=0) / len(dataloader.dataset)

    return mean_product_pre_conv


def compute_product_reshape(input_tensor: torch.Tensor,
                            conv: nn.Conv2d,
                            ) -> torch.Tensor:
    """
    Reshape the product of the product input of a conv2d layer in the shape of the kernel.
    (S x Hd x (2Wd + 1) ) x I -> K x K

    Parameters
    ----------
    input_tensor : torch.Tensor
        the product input of the conv2d layer not reshaped
        A tensor T such that:
        for H > h >= y >= 0  W > w >= z >= 0
        T[s, h, w, x, y, z] = I[x, y, z] * I[s, y + h, z + w]
        where I is the input of the conv2d layer

    conv : nn.Conv2d
        the conv2d layer

    Returns
    -------
    torch.Tensor: K x K
        the product of the input of the conv2d layer in the shape of the kernel
    """

    fpp = torch.nn.functional.pad(
        input_tensor,
        (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0]),
        mode='constant',
        value=0)

    hd, wd = conv.kernel_size
    kernel_input_size = (conv.in_channels, 
                         conv.kernel_size[0],
                         conv.kernel_size[1])
    kp = torch.zeros(kernel_input_size + kernel_input_size, device=input_tensor.device)
    image_shape = fpp.shape[3:]

    for h in range(hd):
        for w in range(wd):
            for y in range(h + 1):
                for z in range(wd):
                    kp[:, h, w, :, y, z] = torch.mean(
                        fpp[:,
                            h - y,
                            w - z + wd,
                            :,
                            y: image_shape[1] + y - hd + 1: conv.stride[0],
                            z: image_shape[2] + z - wd + 1: conv.stride[1]
                            ],
                        dim=(2, 3)
                    )
                    kp[:, y, z, :, h, w] = kp[:, h, w, :, y, z].T
    return kp


def get_mean_product_pre_conv_input(model: nn.Module,
                                    dataloader: DataLoader
                                    ) -> dict[str, torch.Tensor]:
    """
    Get the mean of the product of the input of all the conv2d layers in the shape of the kernel.
    E(X_i Xj) for all i, j in the kernel.

    Parameters
    ----------
    model : nn.Module
        the original PyTorch model
    dataloader : DataLoader
        the dataloader of the dataset

    Returns
    -------
    dict[str, torch.Tensor]: dict[str, K x K]
        the mean of the product of the input of all the conv2d layers
    """
    print("Computing the mean of the product of the input of all the conv2d layers...")
    full_products = get_product_pre_conv_input_full(model, dataloader) 
    print("Reshaping the mean of the product of the input of all the conv2d layers...")
    kernel_mean_products = dict()
    for layer in full_products:
        layer_conv = get_module_by_name(model, layer)
        kernel_mean_products[layer] = compute_product_reshape(
            full_products[layer], layer_conv)
    
    print(f"Mean product is in {kernel_mean_products[layer].device}")
    return kernel_mean_products


def compute_covariance(model: nn.Module,
                       dataloader_mean: DataLoader,
                       dataloader_prod: DataLoader,
                       bias: bool = False
                       ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Compute the covariance of the input of all the conv2d layers in the shape of the kernel.

    Parameters
    ----------
    model : nn.Module
        the original PyTorch model
    dataloader_mean : DataLoader
        the dataloader of the dataset for computing E(X_i) E(X_j)
    dataloader_prod : DataLoader
        the dataloader of the dataset for computing E(X_i X_j)
    bias : bool
        whether to compute the covariance with the bias or not
        bias = False: E(X_i Xj) - E(X_i)E(Xj)
        bias = True: n/(n-1) * (E(X_i Xj) - E(X_i)E(Xj)) with n the number of samples (not the number of images)
        bias = True is not implemented yet

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]
        the covariance of the input of all the conv2d layers and
        the product of the mean of the input of all the conv2d layers
        the mean of the product of the input of all the conv2d layers
    """
    assert not bias, "bias = True is not implemented yet"
    print("Computing E(X_i)E(Xj)...")
    kernel_product_means = get_product_mean_pre_conv_input(model, dataloader_mean)
    print("Computing E(X_i Xj)...")
    kernel_mean_products = get_mean_product_pre_conv_input(model, dataloader_prod)
    print("Computing covariance...")
    covariances = dict()
    for layer in kernel_product_means:
        covariances[layer] = kernel_mean_products[layer] - kernel_product_means[layer]
    return covariances, kernel_product_means, kernel_mean_products


def reshape_cholesky(tensor: torch.Tensor, 
                     layer_name: str = 'unknown'
                     ) -> torch.Tensor:
    """
    Reshape a 6th order tensor to a square matrix and compute the Cholesky decomposition.

    Parameters
    ----------
    tensor : torch.Tensor
        the tensor to reshape and to compute the Cholesky decomposition
    layer_name : str
        the name of the layer

    Returns
    -------
    torch.Tensor: K x K
        the Cholesky decomposition of the tensor
    """
    size = tensor.shape[0] * tensor.shape[1] * tensor.shape[2]
    reshaped_tensor = tensor.reshape(size, size)
    try:
        return torch.linalg.cholesky(reshaped_tensor)
    except RuntimeError:
        print(f"The tensor of layer {layer_name} is not positive definite.")
        diagonal = torch.zeros_like(reshaped_tensor)
        for p in (-9, -8, -7, -6, -5, -4, -3):  # 10 ** -9, 10 ** -8, ..., 10 ** -3
            try:
                diagonal.fill_diagonal_(10 ** p)
                cholesky = torch.linalg.cholesky(reshaped_tensor + diagonal)
                print(f"Added {10 ** p} to the diagonal of the tensor of layer {layer_name}.")
                return cholesky
            except RuntimeError:
                pass
        raise RuntimeError(f"The tensor of layer {layer_name} is not positive definite even with additive diagonal.")
       


class ImageDataset(Dataset):
    """Image dataset loader using image paths, labels, and optional transform."""

    def __init__(self,
                 images: Union[str, Path],
                 labels: Optional[Union[str, Path]] = None,
                 transform: Optional[Union[str, Path]] = None,
                 root: Optional[Union[str, Path]] = None):
        super().__init__()
        self.images_ = self._read_lines(images)
        self.labels_ = self._read_labels(labels) if labels else None
        self.transform_ = self._load_transform(transform) if transform else None
        self.root = root or '.'

    @staticmethod
    def _read_lines(path: Union[str, Path]) -> List[str]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing image list: {path}')
        with open(path) as f:
            return f.read().splitlines()

    @staticmethod
    def _read_labels(path: Union[str, Path]) -> List[int]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing labels file: {path}')
        with open(path) as f:
            return [int(line) for line in f.read().splitlines()]

    @staticmethod
    def _load_transform(path: Union[str, Path]) -> Callable:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing transform file: {path}')
        return torch.load(path, weights_only=False)

    def __len__(self) -> int:
        return len(self.images_)

    def __getitem__(self, index: int) -> Union[Tuple[Image.Image, int], Image.Image]:
        img_path = os.path.join(self.root, self.images_[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform_:
            img = self.transform_(img)
        return (img, self.labels_[index]) if self.labels_ else img

def _main(arguments):
    
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if arguments.batch_size_mean is None:
        arguments.batch_size_mean = arguments.batch_size
    if arguments.batch_size_product is None:
        arguments.batch_size_product = arguments.batch_size
    
    print(f"Batch size when computing E(X_i)E(Xj): {arguments.batch_size_mean}")
    print(f"Batch size when computing E(X_i Xj): {arguments.batch_size_product}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using {arguments.workers} workers.")
    
    # parse the arguments and open the ImageNet dataset
    
    data = ImageDataset(arguments.dataset_files, 
                        arguments.dataset_labels,
                        arguments.transf, arguments.root)
    # check if batch size is valid
    assert len(data) % arguments.batch_size_mean == 0, \
        f"batch_size_mean ({arguments.batch_size_mean}) is not a divisor of the dataset size ({len(data)}))"
    assert len(data) % arguments.batch_size_product == 0, \
        f"batch_size_product ({arguments.batch_size_product}) is not a divisor of the dataset size ({len(data)}))"
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    # is_cuda = False
    print(f"Is CUDA available: {is_cuda}")

    # use a PyTorch dataloader to load the dataset
    dataloader_mean = DataLoader(data,
                                 batch_size=arguments.batch_size_mean,
                                 shuffle=True,
                                 num_workers=arguments.workers,
                                 pin_memory=True if is_cuda else False,
                                 pin_memory_device="cuda")
    if arguments.batch_size_mean != arguments.batch_size_product:
        dataloader_prod = DataLoader(data,
                                     batch_size=arguments.batch_size_product,
                                     shuffle=True,
                                     num_workers=arguments.workers,
                                     pin_memory=True if is_cuda else False,
                                     pin_memory_device="cuda")
    else:
        dataloader_prod = dataloader_mean
    # load the model
    model = torch.load(arguments.model, map_location=device,weights_only=False)
    
    t = perf_counter()
    # compute the covariance
    covariances, product_means, mean_products = compute_covariance(model, dataloader_mean,
                                                                   dataloader_prod, arguments.bias)
    print(f"Time to compute the covariance: {perf_counter() - t:.2f}s")

    print("Saving the covariance...")
    # save the covariance
    torch.save(covariances, f"{arguments.output}_covariance.pt")
    torch.save(product_means, f"{arguments.output}_product_means.pt")
    torch.save(mean_products, f"{arguments.output}_mean_products.pt")

    # compute the Cholesky decomposition
    print("Computing the Cholesky decomposition of the covariances...")
    cholesky_covariances = {layer: reshape_cholesky(cov, layer) for layer, cov in covariances.items()}
    torch.save(cholesky_covariances, f"{arguments.output}_cholesky_covariance.pt")

    print("Computing the Cholesky decomposition of the mean products...")
    cholesky_mean_products = {layer: reshape_cholesky(cov, layer) for layer, cov in mean_products.items()}
    torch.save(cholesky_mean_products, f"{arguments.output}_cholesky_mean_products.pt")
    print("Completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return 0


def _arg_parser():
    parser = argparse.ArgumentParser(
        description=
        'Compute the covariance of the input of all the conv2d layers in the shape of the kernel'
    )
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='the path to the model')
    
    parser.add_argument('--transf',
                        type=str,
                        required=True,
                        help='The path to the sequence of transformations to apply to the input data.')
    parser.add_argument('--root',
                        type=str,
                        required=True,
                        help='The path to the root of the dataset.')
    parser.add_argument('--dataset_files',
                        type=str,
                        required=True,
                        help='The path to the file containing the paths to the dataset.')
    parser.add_argument('--dataset_labels',
                        type=str,
                        required=True,
                        help='The path to the file containing the labels of the dataset.')
    parser.add_argument('--batch-size',
                        metavar='batch_size',
                        type=int,
                        default=1,
                        help='the batch size')
    parser.add_argument('--batch-size-mean',
                        metavar='batch_size_mean',
                        type=int,
                        default=None,
                        help='the batch size when computing E(X_i)E(Xj)')
    parser.add_argument('--batch-size-product',
                        metavar='batch_size_product',
                        type=int,
                        default=None,
                        help='the batch size when computing E(X_i Xj)')
    parser.add_argument('--bias',
                        type=bool,
                        default=False,
                        help='whether to compute the covariance with the bias or not')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='the path to the output file')
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='The number of workers used by the dataloader.')
    
    return parser
if __name__ == '__main__':
    par = _arg_parser()
    args = par.parse_args()
    # run the main code
    _main(args)

