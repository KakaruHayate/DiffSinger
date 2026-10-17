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
Instead, a learnable ``text_bias`` vector stands in for the average missing
text embedding (SoulX's preflow was trained on text + pitch + type summed).

SoulX note_type convention (from preprocess/tools/midi_parser.py):
    1 = SP / rest
    2 = normal lyric note
    3 = slur ("-" extension)
DiffSinger mapping: note_rest=True -> 1, otherwise -> 2 (glide/slur info is
not available at this level and is left to the original MelodyEncoder).

Optimizer routing: only the ConvNeXt pointwise convs (pwconv1/pwconv2) are
dense 2-D block MLPs and go to Muon. Everything else stays on AdamW — the
embeddings, the depthwise conv (AdamWConv1d), the 1-D GRN/text_bias vectors,
and the final projections (AdamWLinear).
"""

from typing import Optional

import torch
import torch.nn as nn

from modules.commons.common_layers import AdamWLinear, AdamWConv1d
from utils.hparams import hparams


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1-D parameters so they are routed to AdamW (not Muon).
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma.view(1, 1, -1) * (x * Nx) + self.beta.view(1, 1, -1) + x


class ConvNeXtV2Block(nn.Module):
    # ref: SoulX-Singer soulxsinger/models/modules/convnext.py (from F5-TTS)
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        # depthwise conv (groups=dim) is not a dense matrix -> AdamW.
        self.dwconv = AdamWConv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # ConvNeXt's pointwise convs act as the block MLP: dense 2-D matrices,
        # deliberately left as plain nn.Linear so Muon optimises them.
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
        # New projection: zero-initialised so that at step 0 the branch is an
        # exact no-op and training starts from the baseline model's behaviour.
        self.out_proj = AdamWLinear(soulx_dim, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Stand-in for SoulX's unused text embedding (see module docstring).
        self.use_text_bias = hparams.get('use_soulx_text_bias', True)
        if self.use_text_bias:
            self.text_bias = nn.Parameter(torch.zeros(soulx_dim))

        # Recover the missing text side from DiffSinger's own phoneme encoder
        # output. It is pooled to note level, detached (no gradient back into
        # the encoder) and projected into SoulX's 512-d text space.
        self.use_text_from_encoder = hparams.get('use_soulx_text_from_encoder', True)
        if self.use_text_from_encoder:
            enc_dim = hparams['hidden_size']
            self.text_norm = nn.LayerNorm(enc_dim)
            self.text_proj = AdamWLinear(enc_dim, soulx_dim)

        self.branch_dropout = hparams.get('soulx_melody_branch_dropout', 0.0)

    @staticmethod
    def pool_phoneme_to_note(encoder_out, mel2ph, mel2note, n_notes: int):
        """
        Average phoneme-level encoder states over the frames of each note.

        :param encoder_out: [B, T_ph, C], phoneme-level encoder output (unpadded)
        :param mel2ph: [B, T_frames], 1-based phoneme index per frame (0 = padding)
        :param mel2note: [B, T_frames], 1-based note index per frame (0 = padding)
        :param n_notes: int, note sequence length
        :return: [B, n_notes, C]
        """
        b, _, c = encoder_out.shape
        enc_pad = torch.nn.functional.pad(encoder_out, [0, 0, 1, 0])
        frame_feat = torch.gather(
            enc_pad, 1, mel2ph[..., None].repeat(1, 1, c)
        )  # [B, T_frames, C]
        valid = (mel2note > 0).to(frame_feat.dtype)
        idx = (mel2note - 1).clamp(min=0)
        note_feat = frame_feat.new_zeros(b, n_notes, c)
        counts = frame_feat.new_zeros(b, n_notes)
        note_feat.scatter_add_(
            1, idx[..., None].repeat(1, 1, c), frame_feat * valid[..., None]
        )
        counts.scatter_add_(1, idx, valid)
        return note_feat / counts.clamp(min=1.0)[..., None]

    def forward(
            self, note_midi, note_rest,
            encoder_out: Optional[torch.Tensor] = None,
            mel2ph: Optional[torch.Tensor] = None,
            mel2note: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :param encoder_out: optional [B, T_ph, C], phoneme-level encoder output
        :param mel2ph: optional [B, T_frames], 1-based
        :param mel2note: optional [B, T_frames], 1-based
        :return: [B, T_n, H]
        """
        # MIDI numbers match SoulX's note_pitch domain directly; pad/rest -> 0.
        note_pitch = note_midi.long().clamp(min=0, max=255)
        # note_type: 1 = SP/rest, 2 = normal lyric note
        note_type = 2 - note_rest.long()
        x = self.note_pitch_encoder(note_pitch) + self.note_type_encoder(note_type)

        text_embed: Optional[torch.Tensor] = None
        if (
            self.use_text_from_encoder
            and encoder_out is not None
            and mel2ph is not None
            and mel2note is not None
        ):
            pooled = self.pool_phoneme_to_note(
                encoder_out.detach(), mel2ph, mel2note, note_midi.shape[1]
            )
            text_embed = self.text_proj(self.text_norm(pooled))
        if self.use_text_bias:
            if text_embed is None:
                text_embed = self.text_bias
            else:
                text_embed = text_embed + self.text_bias
        if text_embed is not None:
            x = x + text_embed

        x = self.preflow(x)
        return self.out_proj(x)

    def load_soulx_checkpoint(self, ckpt_path, strict=True):
        """Load weights extracted from SoulX-Singer's model.pt by
        utils/soulx/extract_soulx_melody_branch.py."""
        state = torch.load(ckpt_path, map_location='cpu')

        # Consume the auxiliary initialisation entries that are not module params.
        text_bias_init = state.pop('text_bias_init', None)

        # GRN params were stored as (1, 1, C) by SoulX; ours are (C,).
        for k in list(state.keys()):
            if k.endswith(('.grn.gamma', '.grn.beta')) and state[k].dim() == 3:
                state[k] = state[k].reshape(-1)

        if text_bias_init is not None and getattr(self, 'text_bias', None) is not None:
            state['text_bias'] = text_bias_init

        missing, unexpected = self.load_state_dict(state, strict=strict)
        return missing, unexpected
