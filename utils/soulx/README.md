# SoulX-Singer melody branch

A parallel melody branch for the DiffSinger **pitch/variance** model, initialised
from [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)'s pretrained
note-level front-end. The original `MelodyEncoder` is left untouched; the branch
output is added to it and only affects the pitch model's conditioning, so the
model's inputs/outputs and phoneme dictionary are unchanged.

## Weights

Download the pruned checkpoint from the release:

<https://github.com/KakaruHayate/DiffSinger/releases/download/soulx-melody-branch-v1/soulx_melody_branch.pt>

Contents (43 tensors, 4.30M params):

| key | shape | note |
|---|---|---|
| `note_pitch_encoder.weight` | 128 x 512 | standard MIDI 0-127 (pruned from 256 rows) |
| `note_type_encoder.weight` | 3 x 512 | 1 = rest, 2 = normal |
| `preflow.*` | 4 x ConvNeXtV2Block | 512d, intermediate 1024 |
| `text_bias_init` | 512 | mean of SoulX text embeddings (consumed at load, not a module param) |

SoulX's phoneme/text embedding is intentionally not included: its
syllable-level (with tone) vocabulary is incompatible with DiffSinger's.

## Configuration

```yaml
use_soulx_melody_branch: true
soulx_melody_branch_ckpt: /path/to/soulx_melody_branch.pt
soulx_melody_branch_dropout: 0.1   # whole-branch dropout during training
soulx_melody_branch_freeze: false  # true: only the new adapter layers train
use_soulx_text_bias: true          # learnable stand-in for SoulX text embed
use_soulx_text_from_encoder: true  # pool encoder_out to notes (stop-gradient)
```

The branch input is built as follows:

- `note_midi`: rounded, clamped to 0-127, zeroed on rests/padding.
- `note_rest`: mapped to SoulX note types (1 = rest, 2 = normal).
- text side: phoneme-level `encoder_out` is expanded by `mel2ph`, averaged per
  note with `mel2note`, `detach`ed (no gradient into the encoder), passed through
  `LayerNorm` + `384 -> 512` projection, then added to `text_bias`.

## Reproducing the checkpoint

```
python utils/soulx/extract_soulx_melody_branch.py \
    --src <SoulX-Singer>/pretrained_models/SoulX-Singer/model.pt \
    --dst soulx_melody_branch.pt
```

The script prunes unused embedding rows and derives `text_bias_init` from the
SoulX text embedding table.

## Optimizer routing

Only the dense ConvNeXt `pwconv1/pwconv2` matrices go to Muon. Embeddings,
the depthwise conv, 1-D GRN vectors and the projection layers use AdamW, so no
non-dense parameter reaches Muon.

## ONNX export

The branch is exported inside `<exp>.pitch.onnx`; the deployment signature is
unchanged. `text_norm`, `text_proj` and the note pooling (`GatherElements` /
`ScatterElements`) are all exportable with opset 17.
