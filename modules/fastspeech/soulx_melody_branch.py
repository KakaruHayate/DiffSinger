"""
Melody branch migrated from SoulX-Singer (https://github.com/Soul-AILab/SoulX-Singer).

This module reuses SoulX-Singer's pretrained note-level front-end:
    note_pitch_encoder (Embedding 256->512)
    note_type_encoder  (Embedding 256->512)
    preflow            (4 x ConvNeXtV2Block(512, 1024))

It runs *in parallel* to DiffSinger's own MelodyEncoder (which stays untouched)
and its output is projected to hidden_size and added to the melody condition.
The phoneme/text embedding of SoulX-Singer is NOT used because its vocabulary
(syllable-level, with tones) is incompatible with DiffSinger's phoneme set.

SoulX note_type convention (from preprocess/tools/midi_parser.py):
    1 = SP / rest
    2 = normal lyric note
    3 = slur ("-" extension)
DiffSinger mapping: note_rest=True -> 1, otherwise -> 2 (glide/slur info is
not available at this level and is left to the original MelodyEncoder).
"""

import torch
import torch.nn as nn

from utils.hparams import hparams


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    # ref: SoulX-Singer soulxsinger/models/modules/convnext.py (from F5-TTS)
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


SOULX_NOTE_TYPE_REST = 1
SOULX_NOTE_TYPE_NORMAL = 2


class SoulXMelodyBranch(nn.Module):
    def __init__(self, soulx_dim=512, soulx_layers=4):
        super().__init__()
        hidden_size = hparams['hidden_size']
        self.note_pitch_encoder = nn.Embedding(256, soulx_dim)
        self.note_type_encoder = nn.Embedding(256, soulx_dim)
        self.preflow = nn.Sequential(
            *[ConvNeXtV2Block(soulx_dim, soulx_dim * 2) for _ in range(soulx_layers)]
        )
        self.out_proj = nn.Linear(soulx_dim, hidden_size)
        self.branch_dropout = hparams.get('soulx_melody_branch_dropout', 0.0)

    def forward(self, note_midi, note_rest):
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :return: [B, T_n, H]
        """
        # MIDI numbers match SoulX's note_pitch domain directly; pad/rest -> 0.
        note_pitch = note_midi.long().clamp(min=0, max=255)
        note_type = torch.where(
            note_rest,
            torch.full_like(note_pitch, SOULX_NOTE_TYPE_REST),
            torch.full_like(note_pitch, SOULX_NOTE_TYPE_NORMAL),
        )
        x = self.note_pitch_encoder(note_pitch) + self.note_type_encoder(note_type)
        x = self.preflow(x)
        return self.out_proj(x)

    def load_soulx_checkpoint(self, ckpt_path, strict=True):
        """Load weights extracted from SoulX-Singer's model.pt by
        utils/soulx/extract_soulx_melody_branch.py."""
        state = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = self.load_state_dict(state, strict=strict)
        return missing, unexpected
