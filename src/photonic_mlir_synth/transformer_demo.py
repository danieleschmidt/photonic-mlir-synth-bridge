"""
TransformerHLSDemo - Photonic HLS generation for transformer attention heads.

Demonstrates how a transformer attention mechanism maps to silicon photonic
hardware using MZI meshes for matrix operations and WDM for parallelism.
"""

from typing import Dict, List, Tuple
import numpy as np

from .hlsgen import PhotonicHLSGen
from .scheduler import MeshScheduler
from .wavelength import WavelengthMultiplexer


class TransformerHLSDemo:
    """
    HLS generator for transformer attention heads on photonic hardware.

    Maps the standard scaled dot-product attention mechanism:
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    to photonic hardware operations using MZI meshes for matrix multiply
    and optical intensity detection for nonlinearity.
    """

    def __init__(self):
        """Initialize the demo with shared HLS generator."""
        self.hlsgen = PhotonicHLSGen()

    def generate_qkv_projection(self, d_model: int, d_head: int) -> str:
        """
        Generate photonic HLS for Q, K, V projection matrices.

        Each projection is an MZI mesh implementing a d_model -> d_head
        unitary (or truncated) linear transform.

        Args:
            d_model: Input embedding dimension
            d_head: Per-head projection dimension

        Returns:
            Verilog-style HLS string for Q, K, V projections
        """
        lines = [
            self.hlsgen.generate_header(),
            f"// QKV Projection: d_model={d_model} → d_head={d_head}",
            f"// Three parallel MZI meshes for Q, K, V",
            f"",
        ]

        for proj_name in ("query", "key", "value"):
            layer_hls = self.hlsgen.generate_layer(
                f"proj_{proj_name}", d_model, d_head
            )
            lines.append(f"// --- {proj_name.upper()} projection ---")
            lines.append(layer_hls)
            lines.append("")

        lines.append(self.hlsgen.generate_footer())
        return "\n".join(lines)

    def generate_scaled_dot_product(self, seq_len: int, d_head: int) -> str:
        """
        Generate photonic HLS for scaled dot-product attention.

        Implements QK^T / sqrt(d_k) using an MZI mesh for the inner product
        and optical detection for the softmax nonlinearity approximation.

        Args:
            seq_len: Sequence length (number of tokens)
            d_head: Per-head dimension

        Returns:
            Verilog-style HLS string for the attention computation
        """
        scale = float(1.0 / np.sqrt(d_head))

        lines = [
            f"// Scaled Dot-Product Attention",
            f"// seq_len={seq_len}, d_head={d_head}, scale={scale:.6f}",
            f"",
            f"module scaled_dot_product_attn_{seq_len}x{d_head} (",
        ]

        # Q, K, V inputs
        for i in range(seq_len):
            lines.append(f"  input  wire [31:0] q_{i} [{d_head-1}:0],")
        for i in range(seq_len):
            lines.append(f"  input  wire [31:0] k_{i} [{d_head-1}:0],")
        for i in range(seq_len):
            lines.append(f"  input  wire [31:0] v_{i} [{d_head-1}:0],")
        for i in range(seq_len):
            sep = "," if i < seq_len - 1 else ""
            lines.append(f"  output wire [31:0] attn_out_{i} [{d_head-1}:0]{sep}")
        lines.append(");")
        lines.append("")

        lines.append(f"  parameter real SCALE = {scale:.8f};  // 1/sqrt(d_head)")
        lines.append("")

        # Compute QK^T scores (seq_len x seq_len)
        lines.append(f"  // Step 1: Compute QK^T attention scores ({seq_len}x{seq_len})")
        lines.append(f"  wire [31:0] scores [{seq_len-1}:0][{seq_len-1}:0];")
        lines.append(f"  genvar qi, ki, di;")
        lines.append(f"  generate")
        lines.append(f"    for (qi = 0; qi < {seq_len}; qi = qi + 1) begin : q_loop")
        lines.append(f"      for (ki = 0; ki < {seq_len}; ki = ki + 1) begin : k_loop")
        lines.append(f"        // Dot product via MZI homodyne detection")
        lines.append(f"        wire [31:0] dot_acc;")
        lines.append(f"        assign dot_acc = 0;")
        lines.append(f"        for (di = 0; di < {d_head}; di = di + 1) begin : d_loop")
        lines.append(f"          // Optical interference: |E_q + E_k|^2 - |E_q - E_k|^2 ∝ Re(E_q · E_k*)")
        lines.append(f"          assign dot_acc = dot_acc + q_qi[di] * k_ki[di];")
        lines.append(f"        end")
        lines.append(f"        assign scores[qi][ki] = SCALE * dot_acc;")
        lines.append(f"      end")
        lines.append(f"    end")
        lines.append(f"  endgenerate")
        lines.append("")

        lines.append(f"  // Step 2: Softmax via optical gain normalization")
        lines.append(f"  wire [31:0] attn_weights [{seq_len-1}:0][{seq_len-1}:0];")
        lines.append(f"  // (Optical softmax: amplified spontaneous emission normalization)")
        lines.append(f"  // softmax approximated by: exp(x) / sum(exp(x))")
        lines.append(f"  // Hardware: nonlinear optical gain + waveguide coupler tree")
        lines.append(f"  generate")
        lines.append(f"    for (qi = 0; qi < {seq_len}; qi = qi + 1) begin : softmax_loop")
        lines.append(f"      wire [31:0] exp_sum;")
        lines.append(f"      assign exp_sum = 0;")
        lines.append(f"      for (ki = 0; ki < {seq_len}; ki = ki + 1) begin : exp_loop")
        lines.append(f"        assign exp_sum = exp_sum + $exp(scores[qi][ki]);")
        lines.append(f"        assign attn_weights[qi][ki] = $exp(scores[qi][ki]);")
        lines.append(f"      end")
        lines.append(f"      for (ki = 0; ki < {seq_len}; ki = ki + 1) begin : norm_loop")
        lines.append(f"        assign attn_weights[qi][ki] = attn_weights[qi][ki] / exp_sum;")
        lines.append(f"      end")
        lines.append(f"    end")
        lines.append(f"  endgenerate")
        lines.append("")

        lines.append(f"  // Step 3: Weighted sum attn_weights @ V")
        lines.append(f"  generate")
        lines.append(f"    for (qi = 0; qi < {seq_len}; qi = qi + 1) begin : out_loop")
        lines.append(f"      for (di = 0; di < {d_head}; di = di + 1) begin : out_d_loop")
        lines.append(f"        assign attn_out_qi[di] = 0;")
        lines.append(f"        for (ki = 0; ki < {seq_len}; ki = ki + 1) begin : out_k_loop")
        lines.append(f"          assign attn_out_qi[di] = attn_out_qi[di] + attn_weights[qi][ki] * v_ki[di];")
        lines.append(f"        end")
        lines.append(f"      end")
        lines.append(f"    end")
        lines.append(f"  endgenerate")
        lines.append("")
        lines.append(f"endmodule  // scaled_dot_product_attn_{seq_len}x{d_head}")

        return "\n".join(lines)

    def generate_attention_head(
        self, seq_len: int, d_model: int, d_head: int
    ) -> str:
        """
        Generate HLS for a complete attention head.

        Combines QKV projection + scaled dot-product attention + WDM assignment.

        Args:
            seq_len: Sequence length
            d_model: Model embedding dimension
            d_head: Per-head dimension

        Returns:
            Verilog-style HLS string for the full attention head
        """
        lines = [
            self.hlsgen.generate_header(),
            f"// Complete Attention Head",
            f"// seq_len={seq_len}, d_model={d_model}, d_head={d_head}",
            f"",
        ]

        # WDM assignment for parallel projections
        wdm = WavelengthMultiplexer(num_wavelengths=4)
        path_ids = list(range(d_head))
        wdm_assign = wdm.assign_wavelengths(path_ids)

        lines.append(f"// WDM channel assignment for {d_head} parallel projection paths")
        for path_id, wavelength in list(wdm_assign.items())[:4]:
            lines.append(f"//   path_{path_id}: λ = {wavelength:.2f} nm")
        if d_head > 4:
            lines.append(f"//   ... (wrapping {d_head} paths over {wdm.channel_count()} WDM channels)")
        lines.append("")

        # QKV projections
        lines.append(self.generate_qkv_projection(d_model, d_head))
        lines.append("")

        # Scaled dot-product attention
        lines.append(self.generate_scaled_dot_product(seq_len, d_head))

        lines.append(self.hlsgen.generate_footer())
        return "\n".join(lines)

    def full_demo(
        self, seq_len: int = 8, d_model: int = 32, d_head: int = 8
    ) -> Dict[str, object]:
        """
        Run a complete attention head HLS generation demo.

        Args:
            seq_len: Sequence length (default: 8)
            d_model: Model dimension (default: 32)
            d_head: Head dimension (default: 8)

        Returns:
            Dict with keys:
              - 'hls': Full HLS output string
              - 'wdm_assignments': Dict of path_id -> wavelength_nm
              - 'pipeline_stages': List of pipeline stages from MeshScheduler
              - 'pipeline_depth': Total pipeline depth
              - 'mzi_count': Total number of MZIs in the mesh
        """
        # Generate full HLS
        hls_output = self.generate_attention_head(seq_len, d_model, d_head)

        # WDM channel assignments
        wdm = WavelengthMultiplexer(num_wavelengths=4)
        path_ids = list(range(d_head))
        wdm_assignments = wdm.assign_wavelengths(path_ids)

        # Pipeline schedule
        scheduler = MeshScheduler(mesh_size=d_model)
        scheduler.schedule_clements(d_model)
        stages = scheduler.get_stages()
        depth = scheduler.get_depth()
        mzi_count = scheduler.get_mzi_count()

        return {
            "hls": hls_output,
            "wdm_assignments": wdm_assignments,
            "pipeline_stages": stages,
            "pipeline_depth": depth,
            "mzi_count": mzi_count,
        }
