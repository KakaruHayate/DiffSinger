"""Extract the SoulX-Singer pretrained melody front-end (note_pitch_encoder,
note_type_encoder, preflow) from model.pt into a small standalone checkpoint
that SoulXMelodyBranch can load.

Usage (diffsinger conda env):
    python utils/soulx/extract_soulx_melody_branch.py \
        --src J:/SoulX-Singer-main/pretrained_models/SoulX-Singer/model.pt \
        --dst J:/soulx_assets/soulx_melody_branch.pt
"""

import argparse
import torch

# SoulXSinger checkpoint key -> SoulXMelodyBranch key
KEY_MAP_PREFIXES = {
    'note_pitch_encoder.': 'note_pitch_encoder.',
    'note_type_encoder.': 'note_type_encoder.',
    'preflow.': 'preflow.',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Path to SoulX-Singer model.pt')
    ap.add_argument('--dst', required=True, help='Output checkpoint path')
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt = ckpt['model']

    out = {}
    for k, v in ckpt.items():
        for prefix in KEY_MAP_PREFIXES:
            if k.startswith(prefix):
                out[k] = v.float() if torch.is_tensor(v) else v
                break

    if not out:
        raise RuntimeError(
            'No melody-branch keys found. Top-level keys sample: '
            + ', '.join(list(ckpt.keys())[:20])
        )

    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in out.values() if torch.is_tensor(v))
    print(f'Extracted {len(out)} tensors, {n_params/1e6:.2f}M params -> {args.dst}')
    for k in sorted(out):
        print(f'  {k}: {tuple(out[k].shape)}')


if __name__ == '__main__':
    main()
