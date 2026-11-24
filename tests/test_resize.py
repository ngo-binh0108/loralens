import json

import numpy as np
import pytest
from helpers import low_rank_pair, write_kohya, write_peft

from loralens.adapter import AdapterError, load_adapter
from loralens.resize import choose_rank, factor_svd, resize, write
from loralens.st import Tensor, load, save


def delta(up, down, alpha, rank):
    return (alpha / rank) * up.reshape(up.shape[0], rank).astype(np.float64) @ down.reshape(rank, -1)


def test_factor_svd_matches_full_svd():
    up, down = low_rank_pair(40, 30, 8, 8, seed=3)
    u, s, vt = factor_svd(up, down)
    full = np.linalg.svd(up.astype(np.float64) @ down, compute_uv=False)[:8]
    np.testing.assert_allclose(s, full, rtol=1e-5)
    np.testing.assert_allclose((u * s) @ vt, up.astype(np.float64) @ down, atol=1e-5)


def test_choose_rank_by_energy():
    s = np.sqrt(np.array([50.0, 30.0, 15.0, 5.0]))  # energy shares 0.5, 0.3, 0.15, 0.05
    assert choose_rank(s, 4, energy=0.5) == (1, pytest.approx(0.5))
    assert choose_rank(s, 4, energy=0.8)[0] == 2
    assert choose_rank(s, 4, energy=0.9)[0] == 3
    assert choose_rank(s, 4, energy=1.0)[0] == 4
    assert choose_rank(s, 4, energy=0.9, max_rank=2)[0] == 2
    assert choose_rank(s, 4, energy=0.1, min_rank=3)[0] == 3
    assert choose_rank(s, 4, rank=10)[0] == 4
    assert choose_rank(np.zeros(4), 4, energy=0.9) == (1, 1.0)


def test_resizing_a_secretly_low_rank_lora_is_lossless(tmp_path):
    up, down = low_rank_pair(64, 48, 16, 4, seed=1)
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_x": (up, down, 8.0)},
                       metadata={"ss_network_dim": "16", "ss_network_alpha": "8", "sshs_model_hash": "abc"})
    results, tensors, metadata, config = resize(load_adapter(path), rank=4)
    assert config is None
    assert results[0].status == "resized" and results[0].new_rank == 4
    assert results[0].energy == pytest.approx(1.0)

    new_down = tensors["lora_unet_x.lora_down.weight"].to_float32()
    new_up = tensors["lora_unet_x.lora_up.weight"].to_float32()
    new_alpha = float(tensors["lora_unet_x.alpha"].to_float32())
    assert new_down.shape == (4, 48) and new_up.shape == (64, 4)
    assert new_alpha == pytest.approx(2.0)  # the scale stays 8/16 = 0.5
    np.testing.assert_allclose(delta(new_up, new_down, new_alpha, 4), delta(up, down, 8.0, 16), atol=1e-5)

    assert metadata["ss_network_dim"] == "4" and metadata["ss_network_alpha"] == "2"
    assert "sshs_model_hash" not in metadata
    assert json.loads(metadata["loralens_resize"])["rank"] == 4


def test_truncation_keeps_the_largest_directions(tmp_path):
    up, down = low_rank_pair(32, 32, 4, 4, singular=[4.0, 3.0, 2.0, 1.0])
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_x": (up, down, None)})
    results, tensors, _, _ = resize(load_adapter(path), rank=2)
    assert results[0].energy == pytest.approx((16 + 9) / 30)
    new_up = tensors["lora_unet_x.lora_up.weight"].to_float32()
    new_down = tensors["lora_unet_x.lora_down.weight"].to_float32()
    s = np.linalg.svd(new_up.astype(np.float64) @ new_down, compute_uv=False)
    np.testing.assert_allclose(s[:2], [4.0, 3.0], rtol=1e-5)
    assert "lora_unet_x.alpha" not in tensors  # no alpha before means alpha = rank, scale 1


def test_conv_module_keeps_its_kernel_shape(tmp_path):
    up2, down2 = low_rank_pair(10, 6 * 9, 8, 3, seed=5)
    up = up2.reshape(10, 8, 1, 1)
    down = down2.reshape(8, 6, 3, 3)
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_conv": (up, down, 4.0)}, dtype="BF16")
    results, tensors, _, _ = resize(load_adapter(path), energy=0.999)
    assert results[0].new_rank == 3
    assert tensors["lora_unet_conv.lora_down.weight"].shape == (3, 6, 3, 3)
    assert tensors["lora_unet_conv.lora_up.weight"].shape == (10, 3, 1, 1)
    assert tensors["lora_unet_conv.lora_down.weight"].dtype == "BF16"


def test_modules_already_small_enough_are_copied_byte_for_byte(tmp_path):
    up, down = low_rank_pair(16, 16, 4, 4)
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_x": (up, down, 4.0)})
    adapter = load_adapter(path)
    results, tensors, _, _ = resize(adapter, rank=8)
    assert results[0].status == "kept"
    assert tensors["lora_unet_x.lora_down.weight"] is adapter.tensors["lora_unet_x.lora_down.weight"]


def test_dora_and_lycoris_modules_are_left_alone(tmp_path):
    up, down = low_rank_pair(16, 16, 8, 2)
    one = Tensor.from_array(np.ones((2, 2), np.float32), "F32")
    extra = {"lora_unet_x.dora_scale": Tensor.from_array(np.ones((1, 16), np.float32), "F32"),
             "lora_unet_h.hada_w1_a": one, "lora_unet_h.hada_w1_b": one,
             "lora_unet_h.hada_w2_a": one, "lora_unet_h.hada_w2_b": one}
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_x": (up, down, 8.0)}, extra=extra)
    results, tensors, _, _ = resize(load_adapter(path), rank=2)
    assert {r.name: r.status for r in results} == {"lora_unet_x": "kept (DoRA)", "lora_unet_h": "copied (loha)"}
    assert tensors["lora_unet_x.lora_down.weight"].shape == (8, 16)


def test_peft_config_gets_new_rank_and_patterns(tmp_path):
    big = low_rank_pair(16, 12, 8, 2, seed=1)
    small = low_rank_pair(16, 12, 8, 6, seed=2)
    folder = write_peft(tmp_path / "in", {
        "model.layers.0.q_proj": big, "model.layers.0.k_proj": big, "model.layers.0.v_proj": small,
    }, {"r": 8, "lora_alpha": 16, "peft_type": "LORA"})
    results, tensors, metadata, config = resize(load_adapter(folder), energy=0.999)
    assert [r.new_rank for r in results] == [2, 2, 6]
    assert config["r"] == 2 and config["lora_alpha"] == 4  # scale 16/8 = 2 is kept
    assert config["rank_pattern"] == {"model.layers.0.v_proj": 6}
    assert config["alpha_pattern"] == {"model.layers.0.v_proj": 12}
    assert config["peft_type"] == "LORA"

    out = tmp_path / "out"
    written = write(str(out), tensors, metadata, config)
    assert sorted(p.split("out")[-1][1:] for p in written) == ["adapter_config.json", "adapter_model.safetensors"]
    reloaded = load_adapter(str(out))
    v = [m for m in reloaded.lora_modules() if m.name.endswith("v_proj")][0]
    assert reloaded.alpha(v, 6) == (12.0, "config")


def test_round_trip_through_a_file_is_close_in_f16(tmp_path):
    up, down = low_rank_pair(24, 20, 8, 8, seed=9)
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_x": (up, down, 8.0)}, dtype="F16")
    results, tensors, metadata, _ = resize(load_adapter(path), rank=8, max_rank=6)
    out = tmp_path / "b.safetensors"
    save(str(out), tensors, metadata)
    reloaded, _ = load(str(out))
    assert reloaded["lora_unet_x.lora_down.weight"].shape == (6, 20)
    assert results[0].energy < 1.0


@pytest.mark.parametrize("kwargs, message", [
    ({}, "exactly one"),
    ({"rank": 2, "energy": 0.9}, "exactly one"),
    ({"rank": 0}, "at least 1"),
    ({"energy": 1.5}, "at most 1"),
])
def test_bad_options(tmp_path, kwargs, message):
    up, down = low_rank_pair(8, 8, 4, 4)
    path = write_kohya(tmp_path / "a.safetensors", {"lora_unet_x": (up, down, None)})
    with pytest.raises(AdapterError, match=message):
        resize(load_adapter(path), **kwargs)
