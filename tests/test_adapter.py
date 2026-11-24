import numpy as np
import pytest
from helpers import low_rank_pair, write_kohya, write_peft

from loralens.adapter import AdapterError, load_adapter, match_pattern
from loralens.info import render, summarize
from loralens.st import Tensor, save


def test_kohya_sdxl_file(tmp_path):
    up, down = low_rank_pair(8, 6, 4, 4)
    path = write_kohya(tmp_path / "style.safetensors", {
        "lora_unet_down_blocks_0_attn1_to_q": (up, down, 2.0),
        "lora_te1_text_model_encoder_layers_0_self_attn_q_proj": (up, down, 2.0),
        "lora_te2_text_model_encoder_layers_0_self_attn_q_proj": (up, down, None),
    }, dtype="F16", metadata={"ss_network_dim": "4", "ss_network_alpha": "2", "sshs_model_hash": "x"})
    adapter = load_adapter(path)
    assert adapter.format == "kohya"
    assert adapter.guess_base() == ("SDXL", "key names")
    summary = summarize(adapter)
    assert summary["modules"] == {"lora": 3}
    assert summary["ranks"] == {4: 3}
    assert summary["alphas"] == {2.0: 2, 4.0: 1}
    assert summary["targets"] == {"lora_unet": 1, "lora_te1": 1, "lora_te2": 1}
    assert "1 module(s) have no alpha" in summary["notes"][0]
    text = render(summary, show_modules=True)
    assert "format   kohya" in text and "8x6" in text
    assert "sshs_model_hash" not in text


def test_peft_folder_reads_alpha_from_config(tmp_path):
    up, down = low_rank_pair(8, 6, 4, 4)
    folder = write_peft(tmp_path / "adapter", {
        "model.layers.0.self_attn.q_proj": (up, down),
        "model.layers.0.self_attn.v_proj": (up, down),
    }, {"r": 4, "lora_alpha": 8, "alpha_pattern": {"v_proj": 2},
        "base_model_name_or_path": "org/tiny-llm"})
    adapter = load_adapter(folder)
    assert adapter.format == "peft"
    assert adapter.guess_base() == ("org/tiny-llm", "adapter_config.json")
    q, v = adapter.lora_modules()
    assert adapter.alpha(q, 4) == (8.0, "config")
    assert adapter.alpha(v, 4) == (2.0, "config")
    assert adapter.scale(8.0, 4) == 2.0


def test_rslora_scale_uses_square_root(tmp_path):
    up, down = low_rank_pair(8, 6, 4, 4)
    folder = write_peft(tmp_path / "a", {"m.q_proj": (up, down)}, {"r": 4, "lora_alpha": 8, "use_rslora": True})
    adapter = load_adapter(folder)
    assert adapter.scale(8.0, 4) == 4.0


def test_peft_pattern_matches_on_a_dot_boundary():
    assert match_pattern({"q_proj": 1}, "model.layers.0.self_attn.q_proj") == 1
    assert match_pattern({"layers.1.self_attn.q_proj": 1}, "model.layers.11.self_attn.q_proj") is None
    assert match_pattern({"proj": 1}, "model.q_proj") is None


def test_lycoris_and_loose_tensors_are_reported(tmp_path):
    one = Tensor.from_array(np.ones((2, 2), np.float32), "F32")
    path = tmp_path / "loha.safetensors"
    save(str(path), {
        "lora_unet_a.hada_w1_a": one, "lora_unet_a.hada_w1_b": one,
        "lora_unet_a.hada_w2_a": one, "lora_unet_a.hada_w2_b": one,
        "lora_unet_a.alpha": Tensor.from_array(np.float32(1), "F32"),
        "emb_params": one,
    })
    adapter = load_adapter(str(path))
    assert adapter.format == "lycoris"
    summary = summarize(adapter)
    assert summary["modules"] == {"loha": 1}
    assert any("loha module(s) can't be resized" in n for n in summary["notes"])
    assert any("1 tensor(s) are not part" in n for n in summary["notes"])


def test_mismatched_rank_is_an_error(tmp_path):
    up, _ = low_rank_pair(8, 6, 4, 4)
    _, down = low_rank_pair(8, 6, 5, 4)
    path = write_kohya(tmp_path / "bad.safetensors", {"lora_unet_x": (up, down, None)})
    with pytest.raises(AdapterError, match="don't share a rank"):
        load_adapter(path)


def test_file_without_lora_tensors_is_an_error(tmp_path):
    path = tmp_path / "model.safetensors"
    save(str(path), {"model.weight": Tensor.from_array(np.ones(2, np.float32), "F32")})
    with pytest.raises(AdapterError, match="no LoRA tensors"):
        load_adapter(str(path))


def test_pickle_and_missing_inputs_are_errors(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "adapter_model.bin").write_bytes(b"x")
    with pytest.raises(AdapterError, match="pickle"):
        load_adapter(str(folder))
    with pytest.raises(AdapterError, match="no such file"):
        load_adapter(str(tmp_path / "missing.safetensors"))


def test_kohya_file_ignores_an_unrelated_sibling_config(tmp_path):
    up, down = low_rank_pair(8, 6, 4, 4)
    (tmp_path / "adapter_config.json").write_text('{"lora_alpha": 99}')
    path = write_kohya(tmp_path / "x.safetensors", {"lora_unet_x": (up, down, None)})
    adapter = load_adapter(path)
    assert adapter.config is None
    assert adapter.alpha(adapter.lora_modules()[0], 4) == (4.0, "default")
