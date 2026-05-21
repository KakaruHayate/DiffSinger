import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms
import yaml
from librosa.filters import mel as librosa_mel_fn

import utils
from .nets import BiLSTMCurveEstimator


class PitchAdjustableMelSpectrogram:
    def __init__(
            self,
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            f_min=40,
            f_max=16000,
            n_mels=128,
            center=False,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center

        self.mel_basis = {}
        self.hann_window = {}

    def __call__(self, y, key_shift=0, speed=1.0):
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * speed))

        mel_basis_key = f"{self.f_max}_{y.device}"
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        hann_window_key = f"{key_shift}_{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(
                win_size_new, device=y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((win_size_new - hop_length) // 2),
                int((win_size_new - hop_length+1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length,
            win_length=win_size_new,
            window=self.hann_window[hann_window_key],
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()

        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))

            spec = spec[:, :size, :] * self.win_size / win_size_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)

        return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class CurveEstimator:
    def __init__(self, model_path: pathlib.Path, device: str):
        if not isinstance(model_path, pathlib.Path):
            model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Curve estimator model path {model_path} does not exist.")
        config_path = model_path.with_name("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Curve estimator config path {config_path} does not exist.")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            dataset_args, model_args = config["dataset_args"], config["model_args"]
        self.device = device
        self.mel_spec_transform = PitchAdjustableMelSpectrogram(
            sample_rate=dataset_args["sample_rate"],
            n_fft=dataset_args["win_size"],
            win_length=dataset_args["win_size"],
            hop_length=dataset_args["hop_size"],
            f_min=dataset_args["f_min"],
            f_max=dataset_args["f_max"],
            n_mels=dataset_args["mel_bins"],
            center=True
        )
        self.model = BiLSTMCurveEstimator(
            **utils.filter_kwargs(model_args, BiLSTMCurveEstimator)
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.model.to(device)
        self.resample_kernels = {}

    @torch.no_grad()
    def estimate(self, waveform: np.ndarray, sr: int, length: int) -> np.ndarray:
        waveform = torch.from_numpy(waveform).float().to(self.device).unsqueeze(0)
        if sr != self.mel_spec_transform.sample_rate:
            if sr not in self.resample_kernels:
                self.resample_kernels[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.mel_spec_transform.sample_rate,
                    lowpass_filter_width=128
                ).to(self.device)
            waveform = self.resample_kernels[sr](waveform)
        mel = dynamic_range_compression_torch(
            self.mel_spec_transform(waveform), clip_val=1e-5
        ).transpose(1, 2)
        pred_curve = self.model.infer(mel).squeeze(0).cpu().numpy()
        pred_curve = np.interp(
            np.linspace(0, len(pred_curve), length),
            np.arange(len(pred_curve)),
            pred_curve
        ).astype(np.float32)
        return pred_curve
