# loralens

[![tests](https://github.com/ngo-binh0108/loralens/actions/workflows/tests.yml/badge.svg)](https://github.com/ngo-binh0108/loralens/actions/workflows/tests.yml)

Look inside a LoRA adapter, and make it smaller.

- `loralens info` shows what a `.safetensors` LoRA really contains: format,
  rank, alpha and the scale they give, target modules, dtype, a guess at the
  base model, and the training metadata.
- `loralens resize` lowers the rank of every LoRA module with a truncated SVD,
  either to a fixed rank or per module to keep a share of the weight change
  (`--energy 95`). Alpha is rewritten so the strength stays the same.

It reads kohya / sd-scripts files (Stable Diffusion, SDXL, Flux), PEFT adapters
(LLMs, with `adapter_config.json`) and diffusers LoRAs. The only dependency is
numpy: no torch and no GPU needed.

## Install

```sh
pip install git+https://github.com/ngo-binh0108/loralens
```

Python 3.10 or newer.

## Inspect

```console
$ loralens info lcm-lora-sdv1-5.safetensors
file     lcm-lora-sdv1-5.safetensors (128.4 MB)
format   kohya
base     SD 1.x / 2.x (from key names)
modules  278 lora
rank     64
alpha    8 (scale 0.125)
dtype    F16
params   67.3M
targets  lora_unet 278
```

`--modules` adds a row per module (rank, alpha, scale and the shape of the
weight it changes), and `--json` prints everything as JSON.

For a PEFT adapter, pass the folder. `alpha` comes from `adapter_config.json`,
including `alpha_pattern` and rsLoRA (`use_rslora`, scale = alpha / sqrt(r)):

```console
$ loralens info ./my-adapter
file     ./my-adapter/adapter_model.safetensors (528.0 KB)
format   peft (adapter_config.json)
base     meta-llama/Llama-3.2-1B (from adapter_config.json)
modules  64 lora
rank     16
alpha    32 (scale 2)
dtype    F32
params   131.1K
targets  q_proj 16, k_proj 16, v_proj 16, o_proj 16
```

`info` also points out things that are easy to miss: modules with no alpha
(they run at scale 1), DoRA magnitudes, LyCORIS modules (LoHa, LoKr, OFT),
and tensors that don't belong to any module.

## Resize

```console
$ loralens resize lcm-lora-sdv1-5.safetensors --energy 95 -o lcm-lora-r95.safetensors
resized  278 of 278 LoRA modules
rank     64 -> 1 to 39
size     128.3 MB -> 9.6 MB
energy   96.5% kept on average, lowest 95.0%
         95.0% at rank 25: lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q
         ...
wrote    lcm-lora-r95.safetensors
```

- `--rank 16` gives every module rank 16 (modules already at 16 or less are
  copied byte for byte).
- `--energy 95` (or `0.95`) gives each module the smallest rank that keeps 95%
  of its weight change. Add `--max-rank` and `--min-rank` to bound it.
- `--dry-run` prints the report without writing, so you can try a few values first.
- A PEFT adapter is written to a new folder, with `r`, `lora_alpha`,
  `rank_pattern` and `alpha_pattern` updated in its `adapter_config.json`.

The output keeps the input's dtypes (F16, BF16, F32) and metadata. kohya's
`ss_network_dim` / `ss_network_alpha` are updated (to `Dynamic` when ranks
differ), stale `sshs_*` hashes are dropped, and a `loralens_resize` entry
records the settings. Existing files are never overwritten without `--force`.

Smaller is not free: always compare a few images or generations before and
after. Energy is a good guide to how much was cut, but it is not a quality score.

## How it works

A LoRA module adds `scale * up @ down` to a base weight, with `up` of shape
(out, r) and `down` of shape (r, in). The rank-k matrix closest to that product
(in the least-squares sense) is its truncated SVD, so that is what `resize`
keeps. Conv modules are flattened to (r, in * kh * kw) and reshaped back.

The product itself is never built. With QR factorisations `up = Qu Ru` and
`down.T = Qd Rd`, the product is `Qu (Ru Rd.T) Qd.T`, and only the small r x r
middle needs an SVD. That makes a module about as cheap to resize as to read,
even at Flux sizes. The new module keeps the old scale, so the new alpha is
`scale * k` (`scale * sqrt(k)` for rsLoRA).

"Energy" is the share of the squared Frobenius norm of the weight change that
the kept singular values hold.

## What it doesn't do

- Only plain LoRA modules are resized. DoRA modules, LoHa, LoKr, OFT, Tucker
  (`lora_mid`) and full `diff` weights are copied unchanged and listed in the report.
- Only `.safetensors`. Pickle files (`.bin`, `.pt`, `.ckpt`) are refused rather
  than loaded, because loading a pickle can run code.
- It doesn't merge a LoRA into a base model or convert between formats.

## Development

```sh
pip install -e ".[test]"
pytest
```

The tests build small LoRA files with a known rank and check the results
exactly, including that a rank-4 product stored at rank 16 shrinks to rank 4
with nothing lost.

## License

MIT
