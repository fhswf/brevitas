from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from torch.nn import Module as TorchModule
from brevitas.nn.mixin import * #WeightQuantType, BiasQuantType
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from typing import Optional, Union
from brevitas.quant_tensor import QuantTensor
from torch.nn import LayerNorm
import torch
from torch.nn.functional import layer_norm

#no idea if QuantWeightBiasInputOutputLayer is applicable for layernorm, but layernorm definitely has a weight, bias and produces an output from an input
class QuantLayerNorm(QuantWBIOL, LayerNorm):
    """
    Self implemented quantized version of Layernorm. It basically is a copy-paste version of QuantLinear, with the inner_forward_impl function adjusted
    to use layernorm. Otherwise, everything else is the same. 

    missing functions from QuantLinear: quant_output_scale_impl, max_acc_bit_width. No idea what they do though
    """
    def __init__(
            self,
            normalized_shape: int,
            eps=1e-05, 
            elementwise_affine=True, 
            bias=True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        LayerNorm.__init__(self, normalized_shape=normalized_shape)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=None,
            output_quant=None,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
    
    def forward(self, input: Union[torch.Tensor, QuantTensor]) -> Union[torch.Tensor, QuantTensor]:
        return self.forward_impl(input)
    
    def inner_forward_impl(self, x: torch.Tensor, quant_weight: torch.Tensor, quant_bias: Optional[torch.Tensor]):
        return layer_norm(x, self.normalized_shape, quant_weight, quant_bias)