"""Optional Triton SiLU-and-multiply kernel for packed Qwen MLP inference.

The input layout is ``[..., 2 * width]``: the first half is the gate and the
second half is the up projection.  Keeping this module separate means the
portable physical-bundle loader never imports Triton unless the user selects
the Triton activation explicitly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    gate_up_ptr,
    output_ptr,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < width
    gate_offsets = row * (2 * width) + columns
    output_offsets = row * width + columns

    gate = tl.load(gate_up_ptr + gate_offsets, mask=mask)
    up = tl.load(gate_up_ptr + gate_offsets + width, mask=mask)
    gate_f32 = gate.to(tl.float32)
    # Match the portable two-op path's dtype boundary as closely as Triton
    # permits: SiLU rounds to the input dtype before the multiply rounds again.
    silu = (gate_f32 * tl.sigmoid(gate_f32)).to(gate.dtype)
    activated = silu.to(tl.float32) * up.to(tl.float32)
    tl.store(output_ptr + output_offsets, activated, mask=mask)


def triton_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """Return fused ``silu(gate) * up`` for a validated packed CUDA tensor."""

    width = gate_up.shape[-1] // 2
    output = torch.empty(
        (*gate_up.shape[:-1], width),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    rows = gate_up.numel() // (2 * width)
    block_size = 256
    grid = (rows, triton.cdiv(width, block_size))
    _silu_and_mul_kernel[grid](
        gate_up,
        output,
        width,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output
