from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEConfig,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

logger = init_logger(__name__)


class MixedPrecisionW4Config(QuantizationConfig):
    """Config class for W4 Mixed Precision quantization.

    This quantization method supports:
    - INT4 weights for expert layers (using TensorRT-LLM unpack)
    - Block INT8 for other layers
    - Mixed precision activation schemes
    """

    def __init__(
        self,
        weight_block_size: list[int] = None,
        ignored_layers: list[str] | None = None,
        int8_channelwise_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_block_size = weight_block_size
        self.ignored_layers = ignored_layers or []
        self.int8_channelwise_layers = int8_channelwise_layers or []

        # W4 quantization parameters
        self.weight_bits = 4
        self.pack_factor = 8 // self.weight_bits

    @classmethod
    def get_name(cls) -> str:
        return "mixed_precision_w4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(
                self.ignored_layers
            )
        if self.int8_channelwise_layers:
            self.int8_channelwise_layers = hf_to_vllm_mapper.apply_list(
                self.int8_channelwise_layers
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MixedPrecisionW4Config":
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"],
                                                 None)
        ignored_layers = cls.get_from_keys_or(
            config, ["ignore", "ignored_layers", "modules_to_not_convert"],
            None)
        int8_channelwise_layers = cls.get_from_keys_or(
            config, ["int8_channelwise_layers"], None)
        return cls(
            weight_block_size=weight_block_size,
            ignored_layers=ignored_layers,
            int8_channelwise_layers=int8_channelwise_layers,
        )

    def get_int8_channelwise_quant_method(self, layer: torch.nn.Module,
                                          prefix: str) -> QuantizeMethodBase:
        config = {
            'config_groups': {
                'group_0': {
                    'input_activations': {
                        'actorder': None,
                        'block_structure': None,
                        'dynamic': True,
                        'group_size': None,
                        'num_bits': 8,
                        'observer': None,
                        'observer_kwargs': {},
                        'strategy': 'token',
                        'symmetric': True,
                        'type': 'int'
                    },
                    'output_activations': None,
                    'targets': ['Linear'],
                    'weights': {
                        'actorder': None,
                        'block_structure': None,
                        'dynamic': False,
                        'group_size': None,
                        'num_bits': 8,
                        'observer': 'minmax',
                        'observer_kwargs': {},
                        'strategy': 'channel',
                        'symmetric': True,
                        'type': 'int'
                    }
                }
            },
            'format': 'int-quantized',
            'global_compression_ratio': None,
            'ignore': [],
            'kv_cache_scheme': None,
            'quant_method': 'compressed-tensors',
            'quantization_status': 'compressed'
        }
        return CompressedTensorsConfig.from_config(config).get_quant_method(
            layer, prefix)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            if should_ignore_layer(prefix,
                                   ignore=self.ignored_layers,
                                   fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            if self.weight_block_size is not None:
                raise NotImplementedError
            else:
                return self.get_int8_channelwise_quant_method(layer, prefix)
        elif isinstance(layer, RoutedExperts):
            if should_ignore_layer(prefix,
                                   ignore=self.int8_channelwise_layers,
                                   fused_mapping=self.packed_modules_mapping):
                return self.get_int8_channelwise_quant_method(layer, prefix)
            logger.info_once(f"Using PPU W4AInt8MoEMethod")
            return W4AInt8MoEMethod(self, layer.moe_config)
        return None


class W4AInt8MoEMethod(FusedMoEMethodBase):
    """MoE method for W4 quantization.
    Supports INT4 weights for expert layers with TensorRT-LLM unpacking.
    """

    def __init__(self, quant_config: MixedPrecisionW4Config,
                 moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.ep_size = moe.ep_size
        self.ep_rank = moe.ep_rank

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # INT4 packed weights for w13 (gate_up_proj) - column parallel
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_size // self.quant_config.pack_factor,
                        dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # INT4 packed weights for w2 (down_proj) - row parallel
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts,
                        hidden_size,
                        intermediate_size_per_partition //
                        self.quant_config.pack_factor,
                        dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts,
                       2 * intermediate_size_per_partition,
                       1,
                       dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """for quant model with acext a8w4 preprocessing"""
        w13_weight = layer.w13_weight
        w13_weight_shape = w13_weight.shape
        w13_weight = w13_weight.view(
            w13_weight_shape[0],
            w13_weight_shape[2] * self.quant_config.pack_factor,
            w13_weight_shape[1] // self.quant_config.pack_factor)
        # [E, hidden_size, 2 * intermediate // 2]
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)

        w2_weight = layer.w2_weight
        w2_weight_shape = w2_weight.shape
        w2_weight = w2_weight.view(
            w2_weight_shape[0],
            w2_weight_shape[2] * self.quant_config.pack_factor,
            w2_weight_shape[1] // self.quant_config.pack_factor)
        # [E, intermediate, hidden_size // 2]
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from acext import (
            pad_to_multiple_of_16,
            get_enum_from_booleans,
            fusedmoe_wrapper,
        )

        assert layer.activation == MoEActivation.SILU, (
            f"Only SiLU activation is supported, not {layer.activation}."
        )

        output = torch.empty_like(x)
        expanded_source_row_to_dest_size = pad_to_multiple_of_16(
            num_tokens=x.shape[0], topk=topk_ids.shape[1])
        Q_type = get_enum_from_booleans(use_fp8_w8a8=False,
                                        use_int8_w8a8=False,
                                        use_int8_w8a16=False,
                                        use_int4_w4a16=False,
                                        use_fp8_w8a16=False,
                                        use_int8_w4a8=True)

        fusedmoe_wrapper(x, layer.w13_weight, layer.w2_weight,
                         topk_weights, topk_ids, output,
                         expanded_source_row_to_dest_size,
                         layer.w13_weight_scale, layer.w2_weight_scale,
                         None, None, None, None, self.ep_rank, self.ep_size, Q_type)

        return output
