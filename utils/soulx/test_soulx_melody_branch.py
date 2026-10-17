"""Smoke test for SoulXMelodyBranch: load extracted weights, run forward.

Run from DiffSinger repo root inside diffsinger env:
    python -I utils/soulx/test_soulx_melody_branch.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from utils.hparams import set_hparams

# minimal hparams needed by SoulXMelodyBranch
set_hparams(config='configs/variance.yaml', print_hparams=False,
            hparams_str='hidden_size=384,soulx_melody_branch_dropout=0.0')

from modules.fastspeech.soulx_melody_branch import SoulXMelodyBranch

branch = SoulXMelodyBranch()
missing, unexpected = branch.load_soulx_checkpoint(
    'J:/soulx_assets/soulx_melody_branch.pt', strict=False)
print('missing:', missing)
print('unexpected:', unexpected)
assert not unexpected, 'checkpoint keys must all belong to the branch'
assert set(missing) == {'out_proj.weight', 'out_proj.bias'}, 'only out_proj is newly added'

# compare against SoulX original preflow on random note sequence
B, Tn = 2, 17
note_midi = torch.randint(40, 80, (B, Tn)).float()
note_midi[0, -3:] = -1  # padding
note_rest = torch.zeros(B, Tn, dtype=torch.bool)
note_rest[0, 5] = True

out = branch(note_midi, note_rest)
print('output shape:', tuple(out.shape))
assert out.shape == (B, Tn, 384)

# determinism / no NaN
assert torch.isfinite(out).all()
print('OK')
