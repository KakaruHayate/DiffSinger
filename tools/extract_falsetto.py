import argparse
import pathlib
import sys

import librosa
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.pe import initialize_pe
from utils.binarizer_utils import get_falsetto_base_harmonic
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams, set_hparams


def main():
    parser = argparse.ArgumentParser(description='Extract H2*/H4* falsetto feature from a waveform.')
    parser.add_argument('wav', type=str, help='Path to the waveform file.')
    parser.add_argument('--out', type=str, default=None, help='Output .npz file path.')
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--win', type=int, default=2048)
    parser.add_argument('--fft', type=int, default=2048)
    parser.add_argument('--device', type=str, default=None, help="Device, e.g. 'cuda' or 'cpu'.")
    args = parser.parse_args()

    set_hparams(config='configs/acoustic.yaml')
    hparams['hop_size'] = args.hop
    hparams['win_size'] = args.win
    hparams['fft_size'] = args.fft
    hparams['pe'] = 'rmvpe'
    hparams['pe_ckpt'] = 'J:/DiffSinger-check-uv-voicing/checkpoints/rmvpe/model.pt'
    hparams['hnsep'] = 'vr'
    hparams['hnsep_ckpt'] = 'J:/DiffSinger-check-uv-voicing/checkpoints/vr/model.pt'
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    waveform, samplerate = librosa.load(args.wav, sr=None, mono=True)
    waveform = waveform.astype(np.float32)
    length = int(np.ceil(len(waveform) / args.hop))

    pe = initialize_pe()
    if hasattr(pe, 'device'):
        pe.device = device
        pe.model.to(device)
        if getattr(pe, 'mel_extractor', None) is not None:
            pe.mel_extractor.to(device)
    f0, uv = pe.get_pitch(
        waveform, samplerate=samplerate, length=length,
        hop_size=args.hop, f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
        interp_uv=False
    )

    dec = DecomposedWaveform(
        waveform, samplerate=samplerate, f0=f0 * ~uv,
        hop_size=args.hop, fft_size=args.fft, win_size=args.win,
        algorithm='vr', device=device
    )

    raw = get_falsetto_base_harmonic(
        dec, None, None, length=length,
        hop_size=args.hop, fft_size=args.fft, win_size=args.win,
        domain='ratio', correction_mode='none'
    )
    corrected = get_falsetto_base_harmonic(
        dec, None, None, length=length,
        hop_size=args.hop, fft_size=args.fft, win_size=args.win,
        domain='ratio', correction_mode='envelope'
    )

    out = args.out or str(pathlib.Path(args.wav).with_suffix('.falsetto.npz'))
    np.savez(out, f0=f0, uv=uv, raw=raw, corrected=corrected)
    print(f'Saved: {out}')
    print(f'raw mean: {raw.mean():.4f}; corrected mean: {corrected.mean():.4f}')
    print(f'raw std: {raw.std():.4f}; corrected std: {corrected.std():.4f}')


if __name__ == '__main__':
    main()
