from typing import Optional, Tuple, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as torchF

import torchsparse.backend
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['conv3d', 'conv3dwbn', 'SparseConvolutionFunction']


class ConvolutionFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(
            ctx,
            input: torch.Tensor,
            weight: torch.Tensor,
            nbmaps: torch.Tensor,
            nbsizes: torch.Tensor,
            sizes: torch.Tensor,
            transposed: bool = False,
    ) -> torch.Tensor:
        input = input.contiguous()
        weight = weight.contiguous()
        nbmaps = nbmaps.contiguous()
        nbsizes = nbsizes.contiguous()

        if not transposed:
            output = torch.zeros(
                sizes[1],
                weight.size(-1),
                dtype=input.dtype,
                device=input.device,
            )
        else:
            # TODO(Haotian): ensure the original, upsampled size to be the same.
            output = torch.zeros(
                sizes[0],
                weight.size(-1),
                dtype=input.dtype,
                device=input.device,
            )

        if input.device.type == 'cuda':
            torchsparse.backend.convolution_forward_cuda(
                input,
                output,
                weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        elif input.device.type == 'cpu':
            torchsparse.backend.convolution_forward_cpu(
                input,
                output,
                weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        else:
            a = 0
            for k in range(weight.shape[0]):
                b = a + nbsizes[k]
                if not transposed:
                    i = nbmaps[a:b, 0].long()
                    o = nbmaps[a:b, 1].long()
                else:
                    i = nbmaps[a:b, 1].long()
                    o = nbmaps[a:b, 0].long()
                output[o] += torch.mm(input[i], weight[k])
                a += nbsizes[k]

        ctx.for_backwards = (input, weight, nbmaps, nbsizes, transposed)
        return output

    @staticmethod
    @custom_bwd
    def backward(
            ctx,
            grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        input, weight, nbmaps, nbsizes, transposed = ctx.for_backwards

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        if grad_output.device.type == 'cuda':
            torchsparse.backend.convolution_backward_cuda(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        elif grad_output.device.type == 'cpu':
            torchsparse.backend.convolution_backward_cpu(
                input,
                grad_input,
                grad_output.contiguous(),
                weight,
                grad_weight,
                nbmaps,
                nbsizes.cpu(),
                transposed,
            )
        else:
            raise NotImplementedError
        return grad_input, grad_weight, None, None, None, None

    @staticmethod
    def symbolic(g, input, weight, nbmaps, nbsizes, sizes, transposed):
        return g.op("ai.onnx.contrib::ConvolutionOnnxFunction", input, weight, nbmaps, nbsizes, sizes,
                    transposed_i=transposed)
        # return g.op("ai.onnx.contrib::ConvolutionOnnxFunction", input, weight, nbmaps, nbsizes, sizes, transposed)


SparseConvolutionFunction = ConvolutionFunction.apply


def conv3d(
        input_feat: torch.Tensor,
        input_coord: torch.Tensor,
        input_cmap: Optional[torch.Tensor] = None,
        input_kmap: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        input_stride: Union[int, Tuple[int, ...]] = 1,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        transposed: bool = False, 
):
    if (kernel_size == 1 and stride == 1 and dilation == 1):
        output_coord = input_coord
        output_feat = input_feat.matmul(weight)
    elif not transposed:
        if input_cmap is None:
            if stride == 1:
                input_cmap = input_coord
            else:
                input_cmap = F.spdownsample(
                    input_coord,
                    stride,
                    kernel_size,
                    input_stride,
                )
        output_coord = input_cmap

        if input_kmap is None:
            offsets = get_kernel_offsets(
                kernel_size,
                stride=input_stride,
                dilation=dilation,
                device=input_feat.device,
            )

            references = F.sphash(input_coord)
            queries = F.sphash(output_coord.int(), offsets)
            input_kmap = F.sphashquery(queries, references)

        nbsizes = torch.sum(input_kmap != -1, dim=1)
        nbmaps = torch.nonzero(input_kmap != -1)
        indices = nbmaps[:, 0] * input_kmap.size(1) + nbmaps[:, 1]
        nbmaps[:, 0] = input_kmap.view(-1)[indices]
        nbmaps = nbmaps.int()
        nbsizes = nbsizes.int()
        sizes = torch.Tensor([input_coord.shape[0], output_coord.shape[0]]).int()
        output_feat = SparseConvolutionFunction(
            input_feat,
            weight,
            nbmaps,
            nbsizes,
            sizes,
            transposed,
        )
    else:
        # when transposed, cmap, kmap must be available
        output_coord = input_cmap
        # TODO(): check runtime efficiency. if consumption expensive, store and
        # use (nbmap, nbsize, size) directly.
        nbsizes = torch.sum(input_kmap != -1, dim=1).int()
        nbmaps = torch.nonzero(input_kmap != -1)
        indices = nbmaps[:, 0] * input_kmap.size(1) + nbmaps[:, 1]
        nbmaps[:, 0] = input_kmap.view(-1)[indices]
        nbmaps = nbmaps.int()
        nbsizes = nbsizes.int()
        sizes = torch.Tensor([output_coord.shape[0], input_coord.shape[0]]).int()
        output_feat = SparseConvolutionFunction(
            input_feat,
            weight,
            nbmaps,
            nbsizes,
            sizes,
            transposed,
        )

    if bias is not None:
        output_feat += bias

    output_cmap = output_coord
    output_kmap = input_kmap

    return output_feat, output_coord, output_cmap, output_kmap


def conv3dwbn(
        input_feat: torch.Tensor,
        input_coord: torch.Tensor,
        input_cmap: Optional[torch.Tensor] = None,
        input_kmap: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        # bn weights
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        bn_weight: Optional[torch.Tensor] = None,
        bn_bias: Optional[torch.Tensor] = None,
        input_stride: Union[int, Tuple[int, ...]] = 1,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        transposed: bool = False,
        is_bn: bool = True,
        is_relu: bool = True,
        bn_training: bool = True,
        exponential_average_factor: float = 0.0,
        eps: float = 1e-5 
):
    if (kernel_size == 1 and stride == 1 and dilation == 1):
        output_coord = input_coord
        output_feat = input_feat.matmul(weight)
    elif not transposed:
        if input_cmap is None:
            if stride == 1:
                input_cmap = input_coord
            else:
                input_cmap = F.spdownsample(
                    input_coord,
                    stride,
                    kernel_size,
                    input_stride,
                )
        output_coord = input_cmap

        if input_kmap is None:
            offsets = get_kernel_offsets(
                kernel_size,
                stride=input_stride,
                dilation=dilation,
                device=input_feat.device,
            )

            references = F.sphash(input_coord)
            queries = F.sphash(output_coord.int(), offsets)
            input_kmap = F.sphashquery(queries, references)

        nbsizes = torch.sum(input_kmap != -1, dim=1)
        nbmaps = torch.nonzero(input_kmap != -1)
        indices = nbmaps[:, 0] * input_kmap.size(1) + nbmaps[:, 1]
        nbmaps[:, 0] = input_kmap.view(-1)[indices]
        nbmaps = nbmaps.int()
        nbsizes = nbsizes.int()
        sizes = torch.Tensor([input_coord.shape[0], output_coord.shape[0]]).int()
        output_feat = SparseConvolutionFunction(
            input_feat,
            weight,
            nbmaps,
            nbsizes,
            sizes,
            transposed,
        )
    else:
        # when transposed, cmap, kmap must be available
        output_coord = input_cmap
        # TODO(): check runtime efficiency. if consumption expensive, store and
        # use (nbmap, nbsize, size) directly.
        nbsizes = torch.sum(input_kmap != -1, dim=1).int()
        nbmaps = torch.nonzero(input_kmap != -1)
        indices = nbmaps[:, 0] * input_kmap.size(1) + nbmaps[:, 1]
        nbmaps[:, 0] = input_kmap.view(-1)[indices]
        nbmaps = nbmaps.int()
        nbsizes = nbsizes.int()
        sizes = torch.Tensor([output_coord.shape[0], input_coord.shape[0]]).int()
        output_feat = SparseConvolutionFunction(
            input_feat,
            weight,
            nbmaps,
            nbsizes,
            sizes,
            transposed,
        )

    if bias is not None:
        output_feat += bias

    output_cmap = output_coord
    output_kmap = input_kmap

    # with bn
    if is_bn:
        output_feat = torchF.batch_norm(
            output_feat,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            bn_training,
            exponential_average_factor,
            eps
        )

    if is_relu:
        output_feat = torchF.relu(output_feat)

    return output_feat, output_coord, output_cmap, output_kmap

