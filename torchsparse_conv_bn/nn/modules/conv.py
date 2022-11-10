
import math
from typing import Optional, Tuple, Union
from torch import Tensor

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as torchF
from torch.nn.parameter import Parameter

from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils import make_ntuple
from torch.autograd import Function


class _Conv3d(Function):

    @staticmethod
    def forward(ctx, input_feat, input_coord, input_cmap, input_kmap, weight, bias, input_stride, kernel_size, stride,
                dilation, transposed):
        return F.conv3d(input_feat,
                        input_coord,
                        input_cmap,
                        input_kmap,
                        weight,
                        bias,
                        input_stride,
                        kernel_size,
                        stride,
                        dilation,
                        transposed)

    @staticmethod
    def symbolic(g, input_feat, input_coord, input_cmap, input_kmap, weight, bias, input_stride,
                 kernel_size, stride, dilation, transposed):
        inputs = [input_feat, input_coord, input_cmap, input_kmap, weight, bias]
        input_w_none = [i for i in inputs if i is not None]

        output_feat, output_coord, output_cmap, output_kmap = g.op("ai.onnx.contrib::Sparseconv", *input_w_none,
                                                                   input_stride_i=input_stride,
                                                                   kernel_size_i=kernel_size, stride_i=stride,
                                                                   dilation_i=dilation, transposed_i=transposed,
                                                                   outputs=4)

        return output_feat, output_coord, output_cmap, output_kmap


class _Conv3dWBN(Function):

    @staticmethod
    def forward(ctx, input_feat, 
                    input_coord, 
                    input_cmap, 
                    input_kmap, 
                    weight, 
                    bias,
                    # bn weights
                    running_mean,
                    running_var,
                    bn_weight,
                    bn_bias, 
                    input_stride, 
                    kernel_size, 
                    stride,
                    dilation, 
                    transposed,
                    # bn params
                    is_bn,
                    is_relu,
                    bn_training,
                    exponential_average_factor,
                    eps):

        return F.conv3dwbn(input_feat,
                        input_coord,
                        input_cmap,
                        input_kmap,
                        weight,
                        bias,
                        running_mean,
                        running_var,
                        bn_weight,
                        bn_bias,
                        input_stride,
                        kernel_size,
                        stride,
                        dilation,
                        transposed,
                        is_bn,
                        is_relu,
                        bn_training,
                        exponential_average_factor,
                        eps)

    @staticmethod
    def symbolic(g, input_feat, 
                    input_coord, 
                    input_cmap, 
                    input_kmap, 
                    weight, 
                    bias,
                    running_mean,
                    running_var,
                    bn_weight,
                    bn_bias, 
                    input_stride,
                    kernel_size, 
                    stride, 
                    dilation, 
                    transposed,
                    is_bn,
                    is_relu,
                    bn_training,
                    exponential_average_factor,
                    eps):
        inputs = [input_feat, input_coord, input_cmap, input_kmap, weight, bias, running_mean, running_var, bn_weight, bn_bias]
        input_w_none = [i for i in inputs if i is not None]

        output_feat, output_coord, output_cmap, output_kmap = g.op("ai.onnx.contrib::Sparseconv", *input_w_none,
                                                                   input_stride_i=input_stride,
                                                                   kernel_size_i=kernel_size, stride_i=stride,
                                                                   dilation_i=dilation, transposed_i=transposed,
                                                                   # - bn params
                                                                   is_bn_i=is_bn, is_relu_i=is_relu, bn_training_i=bn_training,
                                                                   exponential_average_factor_f=exponential_average_factor,
                                                                   eps_f=eps,
                                                                   outputs=4)

        return output_feat, output_coord, output_cmap, output_kmap

class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 input_stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False,
                 is_training: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # make_ntuple(kernel_size, ndim=3)
        self.stride = stride  # make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed
        self.input_stride = input_stride

        self.kernel_volume = int(np.prod(make_ntuple(self.kernel_size, ndim=3)))
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if is_training:
            self.function = F.conv3d
        else:
            self.function = _Conv3d.apply

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input_feat, input_coord, input_cmap, input_kmap):
        return self.function(input_feat,
                             input_coord,
                             input_cmap,
                             input_kmap,
                             self.kernel,
                             self.bias,
                             self.input_stride,
                             self.kernel_size,
                             self.stride,
                             self.dilation,
                             self.transposed)

class Conv3dWBN(nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 input_stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 transposed: bool = False,
                 is_training: bool = True,
                 num_features: int = 0,
                 is_bn: bool = True,
                 is_relu: bool = True,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # make_ntuple(kernel_size, ndim=3)
        self.stride = stride  # make_ntuple(stride, ndim=3)
        self.dilation = dilation
        self.transposed = transposed
        self.input_stride = input_stride

        self.kernel_volume = int(np.prod(make_ntuple(self.kernel_size, ndim=3)))
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # bn + relu
        self.is_bn = is_bn
        self.is_relu = is_relu
        self.init_bn(num_features, eps, momentum, affine, track_running_stats)


        if is_training:
            self.function = F.conv3dwbn
        else:
            self.function = _Conv3dWBN.apply

    def init_bn(self, num_features:int, eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.bn_weight = Parameter(torch.empty(num_features))
            self.bn_bias = Parameter(torch.empty(num_features))
        else:
            self.register_parameter("bn_weight", None)
            self.register_parameter('bn_bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                torch.tensor(0, dtype=torch.long))
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters_bn()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters_bn(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.bn_weight)
            init.zeros_(self.bn_bias)

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )
    
    def bn_forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    self.exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    self.exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            self.bn_training = True
        else:
            self.bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
    
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        if self.is_bn:
            s += ("{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
                "track_running_stats={track_running_stats}".format(**self.__dict__))

        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input_feat, input_coord, input_cmap, input_kmap):
        self.bn_forward(input_feat)

        return self.function(input_feat,
                             input_coord,
                             input_cmap,
                             input_kmap,
                             self.kernel,
                             self.bias,
                             # bn weights
                             self.running_mean if not self.training or self.track_running_stats else None,
                             self.running_var if not self.training or self.track_running_stats else None,
                             self.bn_weight,
                             self.bn_bias,
                             self.input_stride,
                             self.kernel_size,
                             self.stride,
                             self.dilation,
                             self.transposed,
                             self.is_bn,
                             self.is_relu,
                             # bn params
                             self.bn_training,
                             self.exponential_average_factor,
                             self.eps
                             )
