import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


def is_fp8_hardware_supported() -> bool:
    """
    Check if the current device supports FP8 at the hardware level.
    Requires Compute Capability >= 8.9 (Ada Lovelace / RTX 40 series and above).
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9 or (major == 8 and minor >= 9)


class AutoFP8Linear(nn.Module):
    """
    Adaptive FP8 linear layer.
    - Automatically enables FP8 training acceleration when hardware supports it.
    - Gracefully falls back to a standard nn.Linear when hardware does not support FP8.
    - During ONNX export, strips FP8 wrappers to produce a clean high‑precision computation graph.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, layer_name: str = ""):
        super().__init__()

        # Instantiate the standard high‑precision Linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

        # Check hardware and dependency status
        self.use_fp8 = is_fp8_hardware_supported() and TORCHAO_AVAILABLE

        # Optional identifier for logging / debugging
        self.layer_name = layer_name

        config = Float8LinearConfig(
            emulate=False,
        )
        
        # Inject FP8 training logic if supported
        if self.use_fp8:
            # This replaces self.linear parameters with Float8 format supporting delayed scaling
            self.linear = convert_to_float8_training(self.linear, config=config)
            print(
                f"convert_to_float8_training: layer '{self.layer_name}' "
                f"({in_features} -> {out_features})"
            )
            print(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export() and self.use_fp8:
            # torchao may wrap parameters as Float8Tensor or attach scaling factors,
            # which would introduce custom ops (e.g., aten::_scaled_mm).
            # To avoid this, we manually dequantize weights back to the input's dtype.
            
            weight = self.linear.weight
            # Force dequantization by stripping the special tensor state
            if hasattr(weight, "to"):
                weight_high_prec = weight.to(x.dtype)
            else:
                weight_high_prec = weight

            bias_high_prec = None
            if self.linear.bias is not None:
                bias_high_prec = self.linear.bias.to(x.dtype)

            # Standard functional interface – ONNX Tracer recognizes it as MatMul / Gemm
            return F.linear(x, weight_high_prec, bias_high_prec)

        if self.use_fp8:
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            m_tokens = x_2d.shape[0]
            remainder = m_tokens % 16

            if remainder != 0:
                pad_len = 16 - remainder
                x_padded = F.pad(x_2d, (0, 0, 0, pad_len))
                out_padded = self.linear(x_padded)
                out_2d = out_padded[:m_tokens, :]
            else:
                out_2d = self.linear(x_2d)

            return out_2d.view(*orig_shape[:-1], -1)
        
        return self.linear(x)
