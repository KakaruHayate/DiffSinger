from typing import Union

import librosa
import numpy as np
import parselmouth
import torch

from modules.nsf_hifigan.nvSTFT import STFT
from utils.decomposed_waveform import DecomposedWaveform
from utils.pitch_utils import interp_f0


def get_mel_torch(
        waveform, samplerate,
        *,
        num_mel_bins=128, hop_size=512, win_size=2048, fft_size=2048,
        fmin=40, fmax=16000,
        keyshift=0, speed=1, device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stft = STFT(samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax, device=device)
    with torch.no_grad():
        wav_torch = torch.from_numpy(waveform).to(device)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0), keyshift=keyshift, speed=speed).squeeze(0).T
        return mel_torch.cpu().numpy()


@torch.no_grad()
def get_mel2ph_torch(lr, durs, length, timestep, device='cpu'):
    ph_acc = torch.round(torch.cumsum(durs.to(device), dim=0) / timestep + 0.5).long()
    ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(device))
    mel2ph = lr(ph_dur[None])[0]
    num_frames = mel2ph.shape[0]
    if num_frames < length:
        mel2ph = torch.cat((mel2ph, torch.full((length - num_frames,), fill_value=mel2ph[-1], device=device)), dim=0)
    elif num_frames > length:
        mel2ph = mel2ph[:length]
    return mel2ph


def get_pitch_parselmouth(
        waveform, samplerate, length,
        *, hop_size, f0_min=65, f0_max=1100,
        speed=1, interp_uv=False
):
    """

    :param waveform: [T]
    :param samplerate: sampling rate
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param f0_min: Minimum f0 in Hz
    :param f0_max: Maximum f0 in Hz
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hop_size * speed))
    time_step = hop_size / samplerate

    l_pad = int(np.ceil(1.5 / f0_min * samplerate))
    r_pad = hop_size * ((len(waveform) - 1) // hop_size + 1) - len(waveform) + l_pad + 1
    waveform = np.pad(waveform, (l_pad, r_pad))

    # noinspection PyArgumentList
    s = parselmouth.Sound(waveform, sampling_frequency=samplerate).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    )
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array['frequency'].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[: length]
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


def get_energy_librosa(waveform, length, *, hop_size, win_size, domain='db'):
    """
    Definition of energy: RMS of the waveform, in dB representation
    :param waveform: [T]
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param win_size: Window size, in number of samples
    :param domain: db or amplitude
    :return: energy
    """
    energy = librosa.feature.rms(y=waveform, frame_length=win_size, hop_length=hop_size)[0]
    if len(energy) < length:
        energy = np.pad(energy, (0, length - len(energy)))
    energy = energy[: length]
    if domain == 'db':
        energy = librosa.amplitude_to_db(energy, top_db=None)
    elif domain == 'amplitude':
        pass
    else:
        raise ValueError(f'Invalid domain: {domain}')
    return energy


def get_breathiness(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None
):
    """
    Definition of breathiness: RMS of the aperiodic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: breathiness
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_ap = waveform.aperiodic()
    breathiness = get_energy_librosa(
        waveform_ap, length=length,
        hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return breathiness


def get_voicing(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None
):
    """
    Definition of voicing: RMS of the harmonic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: voicing
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_sp = waveform.harmonic()
    voicing = get_energy_librosa(
        waveform_sp, length=length,
        hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return voicing


def get_tension_base_harmonic(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None,
        domain='logit'
):
    """
    Definition of tension: radio of the real harmonic part (harmonic part except the base harmonic)
    to the full harmonic part.
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :return: tension
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_h = waveform.harmonic()
    waveform_base_h = waveform.harmonic(0)
    energy_base_h = get_energy_librosa(
        waveform_base_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size,
        domain='amplitude'
    )
    energy_h = get_energy_librosa(
        waveform_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size,
        domain='amplitude'
    )
    tension = np.sqrt(np.clip(energy_h ** 2 - energy_base_h ** 2, a_min=0, a_max=None)) / (energy_h + 1e-5)
    if domain == 'ratio':
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == 'db':
        tension = np.clip(tension, a_min=1e-5, a_max=1)
        tension = librosa.amplitude_to_db(tension, top_db=None)
    elif domain == 'logit':
        tension = np.clip(tension, a_min=1e-4, a_max=1 - 1e-4)
        tension = np.log(tension / (1 - tension))
    return tension


def get_falsetto_base_harmonic(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None,
        domain='ratio',
        correction_mode='envelope'
):
    """
    Definition of falsetto: the formant-corrected ratio between the second and fourth
    harmonics, `H2* / (H2* + H4*)`. The asterisk indicates that the harmonic amplitudes
    are compensated with the vocal tract response before comparison.
    Refer to: Keating, P. A. (2014). Acoustic Measures of Falsetto Voice. (DOI:10.1121/1.4877544)
    :param waveform: All other analysis parameters will not take effect if a
     DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :param correction_mode: `'envelope'` divides each harmonic energy by the spectral
     envelope value at its frequency; `'none'` keeps the raw ratio for comparison.
    :return: falsetto
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )

    samplerate = waveform.samplerate
    f0 = waveform.f0

    # Harmonic amplitudes are read directly from the STFT magnitude of the separated
    # harmonic part, so no waveform reconstruction and no librosa RMS pass is needed.
    magnitude = waveform.harmonic_spectrum()  # [T_frames, bins]
    fft_size = waveform.fft_size
    bins_per_hz = fft_size / samplerate
    envelope = waveform.spectral_envelope() if correction_mode == 'envelope' else None

    n_frames, n_bins = magnitude.shape
    frame_indices = np.clip(
        np.round(np.linspace(0, len(f0) - 1, length)).astype(int),
        a_min=0, a_max=n_frames - 1
    )
    f0_at_frames = f0[np.clip(frame_indices, 0, len(f0) - 1)].astype(np.float64)
    freq_h2 = 2. * f0_at_frames
    freq_h4 = 4. * f0_at_frames
    bin_h2 = np.clip(np.round(freq_h2 * bins_per_hz).astype(int), 0, n_bins - 1)
    bin_h4 = np.clip(np.round(freq_h4 * bins_per_hz).astype(int), 0, n_bins - 1)
    energy_h2 = magnitude[frame_indices, bin_h2].astype(np.float64)
    energy_h4 = magnitude[frame_indices, bin_h4].astype(np.float64)

    if envelope is not None:
        energy_h2 /= (envelope[frame_indices, bin_h2].astype(np.float64) + 1e-8)
        energy_h4 /= (envelope[frame_indices, bin_h4].astype(np.float64) + 1e-8)
    elif correction_mode != 'none':
        raise ValueError(f'Invalid correction mode: {correction_mode}')

    # Unvoiced frames have no F0, so both harmonic indices collapse onto bin 0 and the
    # ratio would be a meaningless constant of 0.5. Report them as 0 instead.
    denominator = energy_h2 + energy_h4
    voiced = (denominator > 0) & (f0_at_frames > 0)
    falsetto = np.zeros_like(denominator)
    falsetto[voiced] = energy_h2[voiced] / denominator[voiced]
    falsetto = np.clip(falsetto, a_min=0, a_max=1)
    if domain == 'ratio':
        return falsetto
    elif domain == 'db':
        return librosa.amplitude_to_db(np.clip(falsetto, 1e-5, 1), top_db=None)
    elif domain == 'logit':
        falsetto = np.clip(falsetto, a_min=1e-4, a_max=1 - 1e-4)
        return np.log(falsetto / (1 - falsetto))
    else:
        raise ValueError(f'Invalid domain: {domain}')


class SinusoidalSmoothingConv1d(torch.nn.Conv1d):
    def __init__(self, kernel_size):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=max(kernel_size, 1),
            bias=False,
            padding='same',
            padding_mode='replicate'
        )
        if kernel_size > 1:
            smooth_kernel = torch.sin(torch.from_numpy(
                np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi
            ))
            smooth_kernel /= smooth_kernel.sum()
        else:
            smooth_kernel = torch.tensor([1.0], dtype=torch.float32)
        self.weight.data = smooth_kernel[None, None]
