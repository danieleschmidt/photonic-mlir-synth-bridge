"""
Tests for photonic_mlir_synth package.

Covers HLS generation, mesh scheduling, WDM multiplexing,
and the full transformer attention demo.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photonic_mlir_synth import (
    PhotonicHLSGen,
    MeshScheduler,
    WavelengthMultiplexer,
    TransformerHLSDemo,
)


# ─── PhotonicHLSGen tests ────────────────────────────────────────────────────

class TestPhotonicHLSGen:
    def setup_method(self):
        self.gen = PhotonicHLSGen()

    def test_hlsgen_mzi_module(self):
        """MZI module generates expected Verilog structure."""
        hls = self.gen.generate_mzi_module(mzi_id=0, theta=np.pi / 4, phi=np.pi / 3)
        assert "mzi_0" in hls
        assert "module" in hls
        assert "endmodule" in hls
        assert "in_upper" in hls
        assert "out_lower" in hls

    def test_hlsgen_mzi_module_params(self):
        """MZI module embeds theta/phi values."""
        theta = np.pi / 6
        phi = np.pi / 4
        hls = self.gen.generate_mzi_module(mzi_id=5, theta=theta, phi=phi)
        assert "mzi_5" in hls
        # Check that cos/sin values appear
        assert "COS_THETA_HALF" in hls
        assert "SIN_THETA_HALF" in hls
        assert "PHI_PHASE_REAL" in hls

    def test_hlsgen_mesh_module(self):
        """Mesh module generates correct structure for size=4."""
        hls = self.gen.generate_mesh_module(name="test_mesh", size=4)
        assert "test_mesh" in hls
        assert "module" in hls
        assert "endmodule" in hls
        # 4x4 mesh: n_mzi = 4*3//2 = 6
        assert "6 MZI" in hls

    def test_hlsgen_mesh_module_size2(self):
        """Mesh module works for minimal size=2 (1 MZI)."""
        hls = self.gen.generate_mesh_module(name="tiny_mesh", size=2)
        assert "tiny_mesh" in hls
        assert "1 MZI" in hls

    def test_hlsgen_layer(self):
        """Layer module contains expected components."""
        hls = self.gen.generate_layer(layer_name="fc1", in_size=4, out_size=4)
        assert "fc1" in hls
        assert "module" in hls
        assert "endmodule" in hls
        assert "in_0" in hls
        assert "out_0" in hls
        assert "mesh" in hls.lower() or "Mesh" in hls or "mesh" in hls

    def test_hlsgen_layer_different_sizes(self):
        """Layer handles in_size != out_size with padding."""
        hls = self.gen.generate_layer(layer_name="proj", in_size=3, out_size=5)
        assert "proj" in hls
        assert "zero pad" in hls or "pad" in hls.lower()

    def test_hlsgen_output_is_string(self):
        """All generator methods return strings."""
        assert isinstance(self.gen.generate_mzi_module(0, 0.5, 0.3), str)
        assert isinstance(self.gen.generate_mesh_module("m", 3), str)
        assert isinstance(self.gen.generate_layer("l", 2, 2), str)
        assert isinstance(self.gen.generate_header(), str)
        assert isinstance(self.gen.generate_footer(), str)

    def test_hlsgen_header_footer(self):
        """Header and footer contain expected markers."""
        header = self.gen.generate_header()
        footer = self.gen.generate_footer()
        assert "Photonic MLIR" in header
        assert "PhotonicHLSGen" in header
        assert "End of HLS" in footer


# ─── MeshScheduler tests ─────────────────────────────────────────────────────

class TestMeshScheduler:
    def test_scheduler_clements(self):
        """Clements schedule for N=4 has N-1=3 stages."""
        sched = MeshScheduler(mesh_size=4)
        sched.schedule_clements(4)
        stages = sched.get_stages()
        assert len(stages) == 3  # N-1 stages

    def test_scheduler_clements_n6(self):
        """Clements schedule for N=6 has N-1=5 stages."""
        sched = MeshScheduler(mesh_size=6)
        sched.schedule_clements(6)
        assert sched.get_depth() == 5

    def test_scheduler_stages(self):
        """Each stage contains tuples of (row, col) MZI indices."""
        sched = MeshScheduler(mesh_size=4)
        sched.schedule_clements(4)
        stages = sched.get_stages()
        assert all(isinstance(s, list) for s in stages)
        for stage in stages:
            for item in stage:
                assert isinstance(item, tuple)
                assert len(item) == 2

    def test_scheduler_depth(self):
        """Pipeline depth equals N-1 for Clements decomposition."""
        for n in [3, 4, 5, 6, 8]:
            sched = MeshScheduler(mesh_size=n)
            sched.schedule_clements(n)
            assert sched.get_depth() == n - 1, f"Failed for n={n}"

    def test_scheduler_mzi_count(self):
        """Total MZI count is positive and scales with mesh size."""
        for n in [3, 4, 5, 6]:
            sched = MeshScheduler(mesh_size=n)
            sched.schedule_clements(n)
            count = sched.get_mzi_count()
            assert count > 0, f"n={n}: MZI count should be > 0"
            # Count is bounded by n*(n-1)/2 (full Clements) and n-1 (minimum)
            assert count <= n * (n - 1) // 2, f"n={n}: count {count} exceeds max"
            assert count >= (n - 1), f"n={n}: count {count} below minimum"

    def test_scheduler_critical_path(self):
        """Critical path is non-empty and length <= pipeline depth."""
        sched = MeshScheduler(mesh_size=6)
        sched.schedule_clements(6)
        cpath = sched.get_critical_path()
        assert isinstance(cpath, list)
        assert len(cpath) > 0
        assert len(cpath) <= sched.get_depth()

    def test_scheduler_lazy_compute(self):
        """Scheduler computes on first access without explicit schedule call."""
        sched = MeshScheduler(mesh_size=4)
        # Don't call schedule_clements — should still work
        stages = sched.get_stages()
        assert len(stages) == 3

    def test_scheduler_parallelism(self):
        """Parallelism list has one entry per stage."""
        sched = MeshScheduler(mesh_size=5)
        sched.schedule_clements(5)
        par = sched.get_parallelism()
        assert len(par) == sched.get_depth()
        assert all(isinstance(p, int) and p > 0 for p in par)


# ─── WavelengthMultiplexer tests ─────────────────────────────────────────────

class TestWavelengthMultiplexer:
    def test_wavelength_assign(self):
        """Wavelength assignment returns dict with correct keys."""
        wdm = WavelengthMultiplexer(num_wavelengths=4)
        paths = [0, 1, 2, 3]
        result = wdm.assign_wavelengths(paths)
        assert set(result.keys()) == set(paths)

    def test_wavelength_assign_values(self):
        """Assigned wavelengths are within expected range."""
        wdm = WavelengthMultiplexer(num_wavelengths=4, base_wavelength_nm=1550.0, spacing_nm=0.8)
        paths = [0, 1, 2, 3]
        result = wdm.assign_wavelengths(paths)
        for path_id, wl in result.items():
            assert 1550.0 <= wl <= 1550.0 + 3 * 0.8 + 1e-9

    def test_wavelength_get_channel(self):
        """get_channel returns correct wavelength after assignment."""
        wdm = WavelengthMultiplexer(num_wavelengths=4)
        wdm.assign_wavelengths([0, 1, 2, 3])
        wl0 = wdm.get_channel(0)
        wl1 = wdm.get_channel(1)
        assert abs(wl0 - 1550.0) < 1e-9
        assert abs(wl1 - 1550.8) < 1e-9

    def test_wavelength_get_channel_unassigned(self):
        """get_channel raises KeyError for unassigned path."""
        wdm = WavelengthMultiplexer(num_wavelengths=4)
        with pytest.raises(KeyError):
            wdm.get_channel(99)

    def test_wavelength_demultiplex(self):
        """Demultiplex groups paths by wavelength."""
        wdm = WavelengthMultiplexer(num_wavelengths=2)
        # 4 paths over 2 channels: wraps around
        assignments = wdm.assign_wavelengths([0, 1, 2, 3])
        demux = wdm.demultiplex(assignments)
        # Should have 2 wavelengths, each with 2 paths
        assert len(demux) == 2
        for wl, paths in demux.items():
            assert len(paths) == 2

    def test_wavelength_channel_count(self):
        """channel_count returns num_wavelengths."""
        wdm = WavelengthMultiplexer(num_wavelengths=8)
        assert wdm.channel_count() == 8

    def test_wavelength_grid(self):
        """Wavelength grid is correct."""
        wdm = WavelengthMultiplexer(num_wavelengths=3, base_wavelength_nm=1550.0, spacing_nm=1.0)
        expected = np.array([1550.0, 1551.0, 1552.0])
        np.testing.assert_allclose(wdm.wavelengths, expected)


# ─── TransformerHLSDemo tests ─────────────────────────────────────────────────

class TestTransformerHLSDemo:
    def setup_method(self):
        self.demo = TransformerHLSDemo()

    def test_transformer_demo_full(self):
        """full_demo returns dict with expected keys."""
        result = self.demo.full_demo(seq_len=4, d_model=8, d_head=4)
        assert "hls" in result
        assert "wdm_assignments" in result
        assert "pipeline_stages" in result
        assert "pipeline_depth" in result
        assert "mzi_count" in result

    def test_transformer_demo_hls_is_string(self):
        """full_demo HLS output is a non-empty string."""
        result = self.demo.full_demo(seq_len=4, d_model=8, d_head=4)
        assert isinstance(result["hls"], str)
        assert len(result["hls"]) > 100

    def test_transformer_demo_wdm_assignments(self):
        """WDM assignments cover all head dimensions."""
        result = self.demo.full_demo(seq_len=4, d_model=8, d_head=4)
        wdm = result["wdm_assignments"]
        assert len(wdm) == 4  # d_head paths assigned

    def test_transformer_demo_pipeline(self):
        """Pipeline depth is d_model - 1."""
        result = self.demo.full_demo(seq_len=4, d_model=8, d_head=4)
        assert result["pipeline_depth"] == 7  # 8 - 1

    def test_qkv_projection(self):
        """QKV projection generates HLS for all three projections."""
        hls = self.demo.generate_qkv_projection(d_model=8, d_head=4)
        assert "QUERY" in hls or "query" in hls
        assert "KEY" in hls or "key" in hls
        assert "VALUE" in hls or "value" in hls
        assert isinstance(hls, str)

    def test_scaled_dot_product(self):
        """Scaled dot-product attention generates valid HLS."""
        hls = self.demo.generate_scaled_dot_product(seq_len=4, d_head=4)
        assert "scaled_dot_product_attn" in hls
        assert "SCALE" in hls
        assert "softmax" in hls.lower() or "Softmax" in hls or "softmax" in hls

    def test_attention_head_contains_header(self):
        """Full attention head output contains the standard header."""
        hls = self.demo.generate_attention_head(seq_len=4, d_model=8, d_head=4)
        assert "Photonic MLIR" in hls
        assert "PhotonicHLSGen" in hls
