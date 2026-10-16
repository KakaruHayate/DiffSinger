import argparse
import pathlib
import sys

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.pe import initialize_pe
from utils.binarizer_utils import SinusoidalSmoothingConv1d, get_falsetto_base_harmonic
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams, set_hparams


_INIT = {}
_SMOOTHER = SinusoidalSmoothingConv1d(round(0.06 / (512 / 44100))).eval()


def smooth(x):
    with torch.no_grad():
        y = _SMOOTHER(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None])
    return y[0, 0].numpy()


def _init_once():
    if _INIT:
        return _INIT
    set_hparams(config='configs/acoustic.yaml')
    hparams['hop_size'] = 512
    hparams['win_size'] = 2048
    hparams['fft_size'] = 2048
    hparams['pe'] = 'rmvpe'
    hparams['pe_ckpt'] = 'J:/DiffSinger-check-uv-voicing/checkpoints/rmvpe/model.pt'
    hparams['hnsep'] = 'vr'
    hparams['hnsep_ckpt'] = 'J:/DiffSinger-check-uv-voicing/checkpoints/vr/model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pe = initialize_pe()
    if hasattr(pe, 'device'):
        pe.device = device
        pe.model.to(device)
        if getattr(pe, 'mel_extractor', None) is not None:
            pe.mel_extractor.to(device)
    _INIT['pe'] = pe
    _INIT['device'] = device
    return _INIT


def load(wav, seconds):
    ctx = _init_once()
    pe, device = ctx['pe'], ctx['device']
    full, samplerate = librosa.load(wav, sr=None, mono=True)
    n = int(round(seconds * samplerate))
    waveform = full[:n].astype(np.float32)
    length = int(np.ceil(len(waveform) / 512))

    f0, uv = pe.get_pitch(
        waveform, samplerate=samplerate, length=length,
        hop_size=512, f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
        interp_uv=False
    )
    dec = DecomposedWaveform(
        waveform, samplerate=samplerate, f0=f0 * ~uv,
        hop_size=512, fft_size=2048, win_size=2048,
        algorithm='vr', device=device
    )
    v1 = get_falsetto_base_harmonic(
        dec, None, None, length=length,
        hop_size=512, fft_size=2048, win_size=2048,
        domain='ratio', correction_mode='none'
    )
    v2 = get_falsetto_base_harmonic(
        dec, None, None, length=length,
        hop_size=512, fft_size=2048, win_size=2048,
        domain='ratio', correction_mode='envelope'
    )
    return v1, v2, f0, uv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wavs', nargs='+')
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--out', type=str, default='J:/falsettoV2/falsetto_compare.png')
    args = parser.parse_args()

    n = len(args.wavs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, wav in zip(axes, args.wavs):
        v1, v2, f0, uv = load(wav, args.seconds)
        t = np.arange(len(v1)) * 512 / 44100
        v1s, v2s = smooth(v1), smooth(v2)
        ax.plot(t, v1, color='#c0392b', lw=0.6, alpha=0.35)
        ax.plot(t, v2, color='#2471a3', lw=0.6, alpha=0.35)
        ax.plot(t, v1s, color='#c0392b', lw=1.6, label='v1  H2/(H2+H4) raw')
        ax.plot(t, v2s, color='#2471a3', lw=1.6, label='v2  H2*/(H2*+H4*) corrected')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('falsetto ratio')
        name = pathlib.Path(wav).stem
        ax.set_title(f'{name}  (mean v1={v1s.mean():.3f}, v2={v2s.mean():.3f})', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        ax2 = ax.twinx()
        ax2.plot(t, f0, color='#7f8c8d', lw=0.8, alpha=0.55)
        ax2.set_ylabel('f0 (Hz)', fontsize=8)
        ax2.set_ylim(0, 800)
        ax2.tick_params(labelsize=8)

    axes[-1].set_xlabel('time (s)')
    fig.suptitle('Falsetto v1 (raw H2/H4) vs v2 (formant-corrected H2*/H4*)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
