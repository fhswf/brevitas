# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from typing import Optional

import brevitas.config as config
from brevitas.inject.defaults import Int8WeightPerTensorFloat

from .quant_layer import ActQuantType
from .quant_layer import BiasQuantType
from .quant_layer import WeightQuantType
from .quant_scale_bias import QuantScaleBias
from .utils import mul_add_from_bn
from torch import Tensor

class _BatchNormToQuantScaleBias(QuantScaleBias, ABC):

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if running_mean_key in state_dict and running_var_key in state_dict:
            weight_init, bias_init = mul_add_from_bn(
                bn_bias=state_dict[bias_key],
                bn_weight=state_dict[weight_key],
                bn_mean=state_dict[running_mean_key],
                bn_var=state_dict[running_var_key],
                bn_eps=self.eps)
            self.weight.data = weight_init
            self.bias.data = bias_init
            del state_dict[bias_key]
            del state_dict[weight_key]
            del state_dict[running_mean_key]
            del state_dict[running_var_key]
            del state_dict[num_batches_tracked_key]
        super(_BatchNormToQuantScaleBias, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bias_key in missing_keys:
            missing_keys.remove(bias_key)
        if config.IGNORE_MISSING_KEYS and weight_key in missing_keys:
            missing_keys.remove(weight_key)
        if num_batches_tracked_key in unexpected_keys:
            unexpected_keys.remove(num_batches_tracked_key)


class BatchNorm1dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm1dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            runtime_shape=(1, -1, 1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps

    def inner_forward_impl(self, input: Tensor, quant_weight, quant_bias):
        """
        Functionality of BatchNorm, applying a scale and bias.
        Since the implementation of brevitas' batchnorm does not support batches and 
        returns 4d Tensors of shape [1,C,C,E] with C being the features, this function is added.
        It normalizes each sample of a batch and concatenates them into one tensor
        
        Args:
            input: The input to normalize of shape [N,C,L] or [C,L]
            quant_weight: The scale to multiply the input with
            quant_bias: The bias to add to the input
        Returns:
            out: The normalized output of input of shape [N,C,L] or [C,L]
        """
        if len(input.size()) == 2:
            out = super().inner_forward_impl(input, quant_weight, quant_bias)
            return out.squeeze(dim=0)[0]
        N,C,L = input.size()
        #in place modification of tensors fails during gradient computation, unsure why
        samples = input
        out = samples
        #chunking input returns input of shape [1, C, L], BatchNorm1dToQuantScaleBias then returns tensor of shape [1,C,C,L]
        #first dimension can be squeezed together, tensors of second dimension are all the same
        #therefore, take the first "input"
        for i in range(N):
            sample = samples[i].clone()
            x = super().inner_forward_impl(sample, quant_weight, quant_bias)
            #quanttensor value is read only
            out[i] = x.squeeze(dim=0)[0] if isinstance(x, Tensor) else x.value.squeeze(dim=0)[0]
        return out


class BatchNorm2dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm2dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            runtime_shape=(1, -1, 1, 1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps
