import argparse
import pathlib
import sys
import time

import librosa
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules.pe import initialize_pe
from utils.binarizer_utils import get_falsetto_base_harmonic
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams, set_hparams


def main():
    parser = argparse.ArgumentParser(description='Benchmark the falsetto extraction stages.')
    parser.add_argument('wav', type=str, help='Path to the waveform file.')
    parser.add_argument('--seconds', type=float, default=5.0, help='Only analyze this many seconds.')
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--win', type=int, default=2048)
    parser.add_argument('--fft', type=int, default=2048)
    args = parser.parse_args()

    set_hparams(config='configs/acoustic.yaml')
    hparams['hop_size'] = args.hop
    hparams['win_size'] = args.win
    hparams['fft_size'] = args.fft
    hparams['pe'] = 'rmvpe'
    hparams['pe_ckpt'] = 'J:/DiffSinger-check-uv-voicing/checkpoints/rmvpe/model.pt'
    hparams['hnsep'] = 'vr'
    hparams['hnsep_ckpt'] = 'J:/DiffSinger-check-uv-voicing/checkpoints/vr/model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    full, samplerate = librosa.load(args.wav, sr=None, mono=True)
    n = int(round(args.seconds * samplerate))
    waveform = full[:n].astype(np.float32)
    length = int(np.ceil(len(waveform) / args.hop))
    print(f'samples: {len(waveform)} ({args.seconds:.1f}s), frames: {length}')

    t = time.perf_counter()
    pe = initialize_pe()
    if hasattr(pe, 'device'):
        pe.device = device
        pe.model.to(device)
        if getattr(pe, 'mel_extractor', None) is not None:
            pe.mel_extractor.to(device)
    print(f'[rmvpe init] {time.perf_counter() - t:.3f}s')

    t = time.perf_counter()
    f0, uv = pe.get_pitch(
        waveform, samplerate=samplerate, length=length,
        hop_size=args.hop, f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
        interp_uv=False
    )
    torch.cuda.synchronize() if device == 'cuda' else None
    print(f'[rmvpe f0]   {time.perf_counter() - t:.3f}s')

    t = time.perf_counter()
    dec = DecomposedWaveform(
        waveform, samplerate=samplerate, f0=f0 * ~uv,
        hop_size=args.hop, fft_size=args.fft, win_size=args.win,
        algorithm='vr', device=device
    )
    print(f'[vr build]   {time.perf_counter() - t:.3f}s')

    t = time.perf_counter()
    harmonic = dec.harmonic()
    torch.cuda.synchronize() if device == 'cuda' else None
    print(f'[vr split]   {time.perf_counter() - t:.3f}s, samples: {len(harmonic)}')

    t = time.perf_counter()
    h2 = dec.harmonic(1)
    print(f'[h2 istft]   {time.perf_counter() - t:.3f}s')

    t = time.perf_counter()
    h4 = dec.harmonic(3)
    print(f'[h4 istft]   {time.perf_counter() - t:.3f}s')

    t = time.perf_counter()
    envelope = dec.spectral_envelope()
    torch.cuda.synchronize() if device == 'cuda' else None
    print(f'[envelope]   {time.perf_counter() - t:.3f}s, shape: {envelope.shape}')

    t = time.perf_counter()
    raw = get_falsetto_base_harmonic(
        dec, None, None, length=length,
        hop_size=args.hop, fft_size=args.fft, win_size=args.win,
        domain='ratio', correction_mode='none'
    )
    print(f'[raw ratio]  {time.perf_counter() - t:.3f}s, mean: {raw.mean():.4f}')

    t = time.perf_counter()
    corrected = get_falsetto_base_harmonic(
        dec, None, None, length=length,
        hop_size=args.hop, fft_size=args.fft, win_size=args.win,
        domain='ratio', correction_mode='envelope'
    )
    print(f'[corrected]  {time.perf_counter() - t:.3f}s, mean: {corrected.mean():.4f}')


if __name__ == '__main__':
    main()
