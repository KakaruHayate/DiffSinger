"""Extract the SoulX-Singer pretrained melody front-end (note_pitch_encoder,
note_type_encoder, preflow) from model.pt into a small standalone checkpoint
that SoulXMelodyBranch can load.

Usage (diffsinger conda env):
    python utils/soulx/extract_soulx_melody_branch.py \
        --src J:/SoulX-Singer-main/pretrained_models/SoulX-Singer/model.pt \
        --dst J:/soulx_assets/soulx_melody_branch.pt
"""

import argparse
import json

import torch

# SoulXSinger checkpoint key -> SoulXMelodyBranch key
KEY_MAP_PREFIXES = {
    'note_pitch_encoder.': 'note_pitch_encoder.',
    'note_type_encoder.': 'note_type_encoder.',
    'preflow.': 'preflow.',
}

DEFAULT_PHONE_SET = 'J:/SoulX-Singer-main/soulxsinger/utils/phoneme/phone_set.json'


def compute_text_bias(ckpt, phone_set_path):
    """Mean text embedding over real (non-special-token) phoneme rows.

    SoulX's preflow expects text_emb + pitch_emb + type_emb; DiffSinger has no
    compatible text vocabulary, so this mean vector initialises the learnable
    ``text_bias`` stand-in of SoulXMelodyBranch.
    """
    text_weight = ckpt['note_text_encoder.weight']
    num_special = 10  # <PAD>, <SP>, <AP>, <UNK>, <BOW>, <EOW>, <BOS>, <EOS>, <MASK>, <SEP>
    try:
        with open(phone_set_path, encoding='utf8') as f:
            phone_set = json.load(f)
        valid_rows = [
            i for i, p in enumerate(phone_set)
            if i < text_weight.shape[0] and not p.startswith('<')
        ]
        print(f'| text bias: averaging {len(valid_rows)} phoneme rows '
              f'(of {text_weight.shape[0]})')
        return text_weight[valid_rows].mean(dim=0)
    except FileNotFoundError:
        print(f'| phone set not found at {phone_set_path}, '
              f'average rows [{num_special}:] instead')
        return text_weight[num_special:].mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Path to SoulX-Singer model.pt')
    ap.add_argument('--dst', required=True, help='Output checkpoint path')
    ap.add_argument('--phone-set', default=DEFAULT_PHONE_SET,
                    help='SoulX phone_set.json, used for text_bias initialisation')
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

    if 'note_text_encoder.weight' in ckpt:
        out['text_bias_init'] = compute_text_bias(ckpt, args.phone_set).float()

    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in out.values() if torch.is_tensor(v))
    print(f'Extracted {len(out)} tensors, {n_params/1e6:.2f}M params -> {args.dst}')
    for k in sorted(out):
        print(f'  {k}: {tuple(out[k].shape)}')


if __name__ == '__main__':
    main()
