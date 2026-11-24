import json

from helpers import low_rank_pair, write_kohya, write_peft

from loralens.cli import main
from loralens.st import load


def kohya(tmp_path, rank=16, true_rank=4):
    up, down = low_rank_pair(32, 24, rank, true_rank)
    return write_kohya(tmp_path / "style.safetensors", {"lora_unet_a_to_q": (up, down, 8.0),
                                                       "lora_unet_a_to_k": (up, down, 8.0)})


def test_info_text(tmp_path, capsys):
    assert main(["info", kohya(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "format   kohya" in out
    assert "rank     16" in out
    assert "alpha    8 (scale 0.5)" in out


def test_info_json_with_modules(tmp_path, capsys):
    assert main(["info", kohya(tmp_path), "--json", "--modules"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ranks"] == {"16": 2}
    assert [r["name"] for r in data["module_rows"]] == ["lora_unet_a_to_q", "lora_unet_a_to_k"]


def test_resize_writes_a_smaller_file(tmp_path, capsys):
    out = tmp_path / "small.safetensors"
    assert main(["resize", kohya(tmp_path), "-o", str(out), "--energy", "99"]) == 0
    text = capsys.readouterr().out
    assert "resized  2 of 2 LoRA modules" in text
    assert "rank     16 -> 4" in text
    tensors, _ = load(str(out))
    assert tensors["lora_unet_a_to_q.lora_down.weight"].shape == (4, 24)


def test_resize_into_a_folder_keeps_the_file_name(tmp_path, capsys):
    source = kohya(tmp_path)
    folder = tmp_path / "out"
    folder.mkdir()
    assert main(["resize", source, "-o", str(folder), "--rank", "4"]) == 0
    assert (folder / "style.safetensors").exists()


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "small.safetensors"
    assert main(["resize", kohya(tmp_path), "-o", str(out), "--rank", "2", "--dry-run", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["written"] == [] and data["new_ranks"] == {"2": 2}
    assert not out.exists()


def test_refuses_to_overwrite(tmp_path, capsys):
    source = kohya(tmp_path)
    out = tmp_path / "small.safetensors"
    out.write_bytes(b"keep me")
    assert main(["resize", source, "-o", str(out), "--rank", "2"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert out.read_bytes() == b"keep me"
    assert main(["resize", source, "-o", source, "--rank", "2", "--force"]) == 2
    assert "output is the input" in capsys.readouterr().err


def test_peft_output_must_be_a_new_folder(tmp_path, capsys):
    up, down = low_rank_pair(16, 12, 8, 2)
    folder = write_peft(tmp_path / "adapter", {"m.q_proj": (up, down)}, {"r": 8, "lora_alpha": 8})
    assert main(["resize", folder, "-o", folder, "--rank", "2"]) == 2
    assert "new folder" in capsys.readouterr().err
    out = tmp_path / "small"
    assert main(["resize", folder, "-o", str(out), "--rank", "2"]) == 0
    config = json.loads((out / "adapter_config.json").read_text())
    assert config["r"] == 2 and config["lora_alpha"] == 2


def test_errors_are_one_line(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing.safetensors")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("loralens: error: no such file") and "Traceback" not in err
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a file at all")
    assert main(["info", str(bad)]) == 2
    assert "header" in capsys.readouterr().err


def test_resize_needs_an_output_unless_dry_run(tmp_path, capsys):
    assert main(["resize", kohya(tmp_path), "--rank", "2"]) == 2
    assert "-o" in capsys.readouterr().err
