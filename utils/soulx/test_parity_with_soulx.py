"""Verify the ported ConvNeXtV2 stack produces identical outputs to SoulX's
original preflow when fed the same embeddings.

Run from DiffSinger repo root inside diffsinger env:
    python -I utils/soulx/test_parity_with_soulx.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'J:/SoulX-Singer-main')  # only to import the original module definitions

import torch

from utils.hparams import set_hparams
set_hparams(config='configs/variance.yaml', print_hparams=False, hparams_str='hidden_size=384')

from modules.fastspeech.soulx_melody_branch import SoulXMelodyBranch
from soulxsinger.models.modules.convnext import ConvNeXtV2Block as OrigBlock

branch = SoulXMelodyBranch()
branch.load_soulx_checkpoint('J:/soulx_assets/soulx_melody_branch.pt', strict=False)
branch.eval()

# rebuild original preflow and load same weights from model.pt directly
orig = torch.nn.Sequential(*[OrigBlock(512, 1024) for _ in range(4)])
ckpt = torch.load('J:/SoulX-Singer-main/pretrained_models/SoulX-Singer/model.pt', map_location='cpu')
if 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
orig_state = {k[len('preflow.'):]: v for k, v in ckpt.items() if k.startswith('preflow.')}
orig.load_state_dict(orig_state)
orig.eval()

# also compare embedding tables
pitch_w = ckpt['note_pitch_encoder.weight']
type_w = ckpt['note_type_encoder.weight']
assert torch.equal(branch.note_pitch_encoder.weight, pitch_w)
assert torch.equal(branch.note_type_encoder.weight, type_w)
print('embedding tables identical')

x = torch.randn(2, 50, 512)
with torch.no_grad():
    y_new = branch.preflow(x)
    y_orig = orig(x)
diff = (y_new - y_orig).abs().max().item()
print('max abs diff of preflow outputs:', diff)
assert diff < 1e-5, 'ported ConvNeXtV2 must match original numerically'
print('OK - parity confirmed')
