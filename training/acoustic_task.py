import contextlib
import hashlib
import pathlib

import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data
import yaml

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.aux_decoder import build_aux_loss
from modules.losses import DiffusionLoss, RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic, ShallowDiffusionOutput
from modules.vocoders.registry import get_vocoder_cls
from utils.hparams import hparams
from utils.plot import spec_to_figure
from utils.shift_mouth_opening_utils import calculate_shifted_opec, sample_truncated_normal

matplotlib.use('Agg')

# Enable TF32 for cuBLAS and cuDNN on Ampere+ GPUs (RTX 4090, 5090, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class AcousticDataset(BaseDataset):
    def __init__(self, prefix, preload=False):
        super(AcousticDataset, self).__init__(prefix, hparams['dataset_size_key'], preload)
        self.required_variances = {}  # key: variance name, value: padding value
        if hparams['use_energy_embed']:
            self.required_variances['energy'] = 0.0
        if hparams['use_breathiness_embed']:
            self.required_variances['breathiness'] = 0.0
        if hparams['use_voicing_embed']:
            self.required_variances['voicing'] = 0.0
        if hparams['use_tension_embed']:
            self.required_variances['tension'] = 0.0
        if hparams['use_mouth_opening_embed']:
            self.required_variances['mouth_opening'] = 0.0

        self.need_key_shift = hparams['use_key_shift_embed']
        self.need_speed = hparams['use_speed_embed']
        self.need_mouth_opening_gt = hparams.get('use_shift_mouth_opening_embed', False)
        self.need_spk_id = hparams['use_spk_id']
        self.need_lang_id = hparams['use_lang_id']

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        f0 = utils.collate_nd([s['f0'] for s in samples], 0.0)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0)
        mel = utils.collate_nd([s['mel'] for s in samples], 0.0)
        batch.update({
            'tokens': tokens,
            'mel2ph': mel2ph,
            'mel': mel,
            'f0': f0,
        })
        for v_name, v_pad in self.required_variances.items():
            batch[v_name] = utils.collate_nd([s[v_name] for s in samples], v_pad)
        if self.need_key_shift:
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])[:, None]
        if self.need_speed:
            batch['speed'] = torch.FloatTensor([s['speed'] for s in samples])[:, None]
        if self.need_mouth_opening_gt:
            batch['mouth_opening_gt'] = utils.collate_nd(
                [s['mouth_opening'] for s in samples], 0.0
            )
        if self.need_spk_id:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if self.need_lang_id:
            languages = utils.collate_nd([s['languages'] for s in samples], 0)
            batch['languages'] = languages
        return batch


class AcousticTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = AcousticDataset
        self.diffusion_type = hparams['diffusion_type']
        assert self.diffusion_type in ['ddpm', 'reflow'], f"Unknown diffusion type: {self.diffusion_type}"
        self.use_shallow_diffusion = hparams['use_shallow_diffusion']
        if self.use_shallow_diffusion:
            self.shallow_args = hparams['shallow_diffusion_args']
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']

        self.use_vocoder = hparams['infer'] or hparams['val_with_vocoder']
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.logged_gt_wav = set()
        self.required_variances = []
        if hparams['use_energy_embed']:
            self.required_variances.append('energy')
        if hparams['use_breathiness_embed']:
            self.required_variances.append('breathiness')
        if hparams['use_voicing_embed']:
            self.required_variances.append('voicing')
        if hparams['use_tension_embed']:
            self.required_variances.append('tension')
        if hparams['use_mouth_opening_embed']:
            self.required_variances.append('mouth_opening')

        self.use_shift_mouth_opening = hparams.get('use_shift_mouth_opening_embed', False)
        if self.use_shift_mouth_opening:
            assert not hparams['use_mouth_opening_embed'], \
                'use_shift_mouth_opening_embed 与 use_mouth_opening_embed 互斥，不能同时启用'
            smo_args = hparams['shift_mouth_opening_args']
            self.smo_alpha_sigma = float(smo_args['alpha_sigma'])
            self.smo_replacement_prob = float(smo_args['replacement_prob'])
            self.smo_o_min = float(smo_args['opec_min'])
            self.smo_o_max = float(smo_args['opec_max'])
            self.teacher_use_amp = bool(smo_args.get('teacher_use_amp', False))
            self.teacher_ckpt_path = pathlib.Path(smo_args['teacher_ckpt_path'])
            assert self.teacher_ckpt_path.is_file(), \
                f'teacher_ckpt_path 不存在: {self.teacher_ckpt_path}'
            self.teacher_dir = self.teacher_ckpt_path.parent
            teacher_config_path = self.teacher_dir / 'config.yaml'
            assert teacher_config_path.is_file(), \
                f'teacher config.yaml 不存在: {teacher_config_path}'
            with open(teacher_config_path, 'r', encoding='utf-8') as f:
                self.teacher_hparams = yaml.safe_load(f)
            self._validate_teacher_hparams(self.teacher_hparams)
            self.teacher: DiffSingerAcoustic = self._build_teacher()
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            self.teacher = None

        super()._finish_init()

        # ── Fuse LYNXNet2 backbone kernels (in-place) ──
        if hparams.get('use_fused_kernels', False):
            from modules.kernels.integration import patch_diffusion_module
            from lightning.pytorch.utilities.rank_zero import rank_zero_info
            n = patch_diffusion_module(
                self.model.diffusion,
                glu_type=hparams['backbone_args'].get('glu_type', 'softsign_glu'),
            )
            rank_zero_info('Fused kernels: patched %d LYNXNet2 blocks', n)

    def _build_model(self):
        return DiffSingerAcoustic(
            vocab_size=len(self.phoneme_dictionary),
            out_dims=hparams['audio_num_mel_bins']
        )

    @staticmethod
    def _hash_file(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    def _validate_teacher_hparams(self, t_hp: dict):
        mel_keys = ['audio_sample_rate', 'hop_size', 'win_size', 'fft_size',
                    'audio_num_mel_bins', 'fmin', 'fmax', 'mel_base']
        for k in mel_keys:
            assert t_hp.get(k) == hparams.get(k), \
                f'teacher / student mel feature mismatch on {k!r}: ' \
                f'teacher={t_hp.get(k)} vs student={hparams.get(k)}'

        for k in ['use_lang_id', 'num_lang', 'use_spk_id', 'num_spk']:
            assert t_hp.get(k) == hparams.get(k), \
                f'teacher / student data space mismatch on {k!r}: ' \
                f'teacher={t_hp.get(k)} vs student={hparams.get(k)}'

        s_dicts = hparams.get('dictionaries') or {'default': hparams.get('dictionary')}
        t_dicts = t_hp.get('dictionaries') or {'default': t_hp.get('dictionary')}
        assert sorted(s_dicts.keys()) == sorted(t_dicts.keys()), \
            f'teacher / student dictionary languages mismatch: ' \
            f'teacher={sorted(t_dicts.keys())} vs student={sorted(s_dicts.keys())}'
        for lang in s_dicts:
            s_path = pathlib.Path(s_dicts[lang])
            t_path = pathlib.Path(t_dicts[lang])
            if not t_path.is_absolute():
                t_path = self.teacher_dir / t_path
            t_dict_in_workdir = self.teacher_dir / (
                f'dictionary-{lang}.txt' if 'dictionaries' in t_hp else 'dictionary.txt'
            )
            if t_dict_in_workdir.is_file():
                t_path = t_dict_in_workdir
            assert s_path.is_file(), f'student dictionary missing: {s_path}'
            assert t_path.is_file(), f'teacher dictionary missing: {t_path}'
            assert self._hash_file(s_path) == self._hash_file(t_path), \
                f'teacher / student dictionary content mismatch for lang {lang!r}'

        s_extra = hparams.get('extra_phonemes') or []
        t_extra = t_hp.get('extra_phonemes') or []
        assert list(s_extra) == list(t_extra), \
            'teacher / student extra_phonemes mismatch'
        s_merged = hparams.get('merged_phoneme_groups') or []
        t_merged = t_hp.get('merged_phoneme_groups') or []
        assert list(s_merged) == list(t_merged), \
            'teacher / student merged_phoneme_groups mismatch'

        assert t_hp.get('use_mouth_opening_embed', False) is True, \
            'teacher must have use_mouth_opening_embed=True'
        for k in ['use_energy_embed', 'use_breathiness_embed', 'use_voicing_embed',
                  'use_tension_embed', 'use_shift_mouth_opening_embed']:
            assert t_hp.get(k, False) is False, \
                f'teacher must NOT enable {k!r}'

    @contextlib.contextmanager
    def _patched_hparams(self, new_hparams: dict):
        saved = dict(hparams)
        try:
            hparams.clear()
            hparams.update(new_hparams)
            hparams['infer'] = saved.get('infer', False)
            yield
        finally:
            hparams.clear()
            hparams.update(saved)

    def _build_teacher(self) -> DiffSingerAcoustic:
        with self._patched_hparams(self.teacher_hparams):
            teacher = DiffSingerAcoustic(
                vocab_size=len(self.phoneme_dictionary),
                out_dims=self.teacher_hparams['audio_num_mel_bins']
            )
            utils.load_ckpt(
                teacher, self.teacher_ckpt_path,
                ckpt_steps=None, prefix_in_ckpt='model', strict=True,
                device='cpu'
            )
            teacher.check_category('acoustic')
        return teacher

    def _teacher_forward(self, sample, shifted_mouth_opening, replace_mask):
        if not replace_mask.any():
            return None
        idx = torch.nonzero(replace_mask, as_tuple=False).squeeze(-1)
        B = replace_mask.shape[0]

        def _slice(v):
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B:
                return v[idx]
            return v

        sub_smo = shifted_mouth_opening[idx]
        with torch.no_grad(), self._patched_hparams(self.teacher_hparams):
            t_variances = {'mouth_opening': sub_smo}
            t_key_shift = _slice(sample.get('key_shift')) if self.teacher_hparams.get('use_key_shift_embed') else None
            t_speed = _slice(sample.get('speed')) if self.teacher_hparams.get('use_speed_embed') else None
            t_spk = _slice(sample.get('spk_ids')) if self.teacher_hparams.get('use_spk_id') else None
            t_lang = _slice(sample.get('languages')) if self.teacher_hparams.get('use_lang_id') else None
            amp_ctx = (
                torch.autocast(device_type=sample['mel'].device.type, dtype=torch.float16)
                if self.teacher_use_amp and sample['mel'].is_cuda
                else contextlib.nullcontext()
            )
            with amp_ctx:
                t_out: ShallowDiffusionOutput = self.teacher(
                    _slice(sample['tokens']),
                    mel2ph=_slice(sample['mel2ph']),
                    f0=_slice(sample['f0']),
                    **t_variances,
                    key_shift=t_key_shift, speed=t_speed,
                    spk_embed_id=t_spk, languages=t_lang,
                    gt_mel=None, infer=True,
                )
        full = sample['mel'].clone()
        full[idx] = t_out.diff_out.to(full.dtype)
        return full

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        if self.use_shallow_diffusion:
            self.aux_mel_loss = build_aux_loss(self.shallow_args['aux_decoder_arch'])
            self.lambda_aux_mel_loss = hparams['lambda_aux_mel_loss']
            self.register_validation_loss('aux_mel_loss')
        if self.diffusion_type == 'ddpm':
            self.mel_loss = DiffusionLoss(loss_type=hparams['main_loss_type'])
        elif self.diffusion_type == 'reflow':
            self.mel_loss = RectifiedFlowLoss(
                loss_type=hparams['main_loss_type'], log_norm=hparams['main_loss_log_norm']
            )
        else:
            raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
        self.register_validation_loss('mel_loss')

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        target = sample['mel']  # [B, T_s, M]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')

        if hparams['use_spk_id']:
            spk_embed_id = sample['spk_ids']
        else:
            spk_embed_id = None
        if hparams['use_lang_id']:
            languages = sample['languages']
        else:
            languages = None

        shift_mouth_opening = None
        if self.use_shift_mouth_opening and not infer:
            B = target.shape[0]
            device = target.device
            # sample alpha from truncated normal
            alpha = sample_truncated_normal(
                B, self.smo_alpha_sigma,
                lo=-1.0, hi=1.0,
                device=device,
                generator=None
            ).to(target.dtype)
            # decide which items are replaced by teacher output
            replace_mask = torch.rand(B, device=device) < self.smo_replacement_prob
            # build frame-level shift_mouth_opening tensor
            mouth_opening_gt = sample['mouth_opening_gt'].to(device=device, dtype=target.dtype)
            shift_mouth_opening = alpha[:, None].expand_as(target[:, :, 0])
            shifted_mouth_opening = calculate_shifted_opec(
                mouth_opening_gt, shift_mouth_opening,
                o_min=self.smo_o_min, o_max=self.smo_o_max,
            )
            # teacher replacement of GT mel target
            new_target = self._teacher_forward(sample, shifted_mouth_opening, replace_mask)
            if new_target is not None:
                target = new_target
        output: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=mel2ph, f0=f0, **variances,
            key_shift=key_shift, speed=speed,
            shift_mouth_opening=shift_mouth_opening,
            spk_embed_id=spk_embed_id, languages=languages,
            gt_mel=target, infer=infer
        )

        if infer:
            return output
        else:
            losses = {}

            if output.aux_out is not None:
                aux_out = output.aux_out
                norm_gt = self.model.aux_decoder.norm_spec(target)
                aux_mel_loss = self.lambda_aux_mel_loss * self.aux_mel_loss(aux_out, norm_gt)
                losses['aux_mel_loss'] = aux_mel_loss

            non_padding = (mel2ph > 0).unsqueeze(-1).float()
            if output.diff_out is not None:
                if self.diffusion_type == 'ddpm':
                    x_recon, x_noise = output.diff_out
                    mel_loss = self.mel_loss(x_recon, x_noise, non_padding=non_padding)
                elif self.diffusion_type == 'reflow':
                    v_pred, v_gt, t = output.diff_out
                    mel_loss = self.mel_loss(v_pred, v_gt, t=t, non_padding=non_padding)
                else:
                    raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
                losses['mel_loss'] = mel_loss

            return losses

    def on_train_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)
        if self.teacher is not None and next(self.teacher.parameters()).device != self.device:
            self.teacher.to(self.device)

    def _on_validation_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)
        if self.teacher is not None and next(self.teacher.parameters()).device != self.device:
            self.teacher.to(self.device)

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        if sample['size'] > 0 and min(sample['indices']) < hparams['num_valid_plots']:
            mel_out: ShallowDiffusionOutput = self.run_model(sample, infer=True)
            for i in range(len(sample['indices'])):
                data_idx = sample['indices'][i].item()
                if data_idx < hparams['num_valid_plots']:
                    if self.use_vocoder:
                        self.plot_wav(
                            data_idx, sample['mel'][i],
                            mel_out.aux_out[i] if mel_out.aux_out is not None else None,
                            mel_out.diff_out[i],
                            sample['f0'][i]
                        )
                    if mel_out.aux_out is not None:
                        self.plot_mel(data_idx, sample['mel'][i], mel_out.aux_out[i], 'auxmel')
                    if mel_out.diff_out is not None:
                        self.plot_mel(data_idx, sample['mel'][i], mel_out.diff_out[i], 'diffmel')
        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_wav(self, data_idx, gt_mel, aux_mel, diff_mel, f0):
        f0_len = self.valid_dataset.metadata['f0'][data_idx]
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        gt_mel = gt_mel[:mel_len].unsqueeze(0)
        if aux_mel is not None:
            aux_mel = aux_mel[:mel_len].unsqueeze(0)
        if diff_mel is not None:
            diff_mel = diff_mel[:mel_len].unsqueeze(0)
        f0 = f0[:f0_len].unsqueeze(0)
        if data_idx not in self.logged_gt_wav:
            gt_wav = self.vocoder.spec2wav_torch(gt_mel, f0=f0)
            self.logger.all_rank_experiment.add_audio(
                f'gt_{data_idx}', gt_wav,
                sample_rate=hparams['audio_sample_rate'],
                global_step=self.global_step
            )
            self.logged_gt_wav.add(data_idx)
        if aux_mel is not None:
            aux_wav = self.vocoder.spec2wav_torch(aux_mel, f0=f0)
            self.logger.all_rank_experiment.add_audio(
                f'aux_{data_idx}', aux_wav,
                sample_rate=hparams['audio_sample_rate'],
                global_step=self.global_step
            )
        if diff_mel is not None:
            diff_wav = self.vocoder.spec2wav_torch(diff_mel, f0=f0)
            self.logger.all_rank_experiment.add_audio(
                f'diff_{data_idx}', diff_wav,
                sample_rate=hparams['audio_sample_rate'],
                global_step=self.global_step
            )

    def plot_mel(self, data_idx, gt_spec, out_spec, name_prefix='mel'):
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        mel_len = self.valid_dataset.metadata['mel'][data_idx]
        spec_cat = torch.cat([(out_spec - gt_spec).abs() + vmin, gt_spec, out_spec], -1)
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(f'{name_prefix}_{data_idx}', spec_to_figure(
            spec_cat[:mel_len], vmin, vmax, title_text
        ), global_step=self.global_step)
