from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import modules.compat as compat
from basics.base_module import CategorizedModule
from modules.aux_decoder import AuxDecoderAdaptor
from modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
    NormalInitEmbedding as Embedding
)
from modules.core import (
    GaussianDiffusion, PitchDiffusion, MultiVarianceDiffusion,
    RectifiedFlow, PitchRectifiedFlow, MultiVarianceRectifiedFlow
)
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.param_adaptor import ParameterAdaptorModule
from modules.fastspeech.tts_modules import RhythmRegulator, LengthRegulator
from modules.fastspeech.variance_encoder import FastSpeech2Variance, MelodyEncoder
from modules.gst.style_encoder import TPSE, StyleEncoder
from utils.hparams import hparams


class ShallowDiffusionOutput:
    def __init__(self, *, aux_out=None, diff_out=None, gst_output=None, tpse_output=None):
        self.aux_out = aux_out
        self.diff_out = diff_out
        self.gst_output = gst_output
        self.tpse_output = tpse_output


class DiffSingerAcoustic(CategorizedModule, ParameterAdaptorModule):
    @property
    def category(self):
        return 'acoustic'

    def __init__(self, vocab_size, out_dims):
        CategorizedModule.__init__(self)
        ParameterAdaptorModule.__init__(self)
        self.fs2 = FastSpeech2Acoustic(
            vocab_size=vocab_size
        )

        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        self.shallow_args = hparams.get('shallow_diffusion_args', {})
        if self.use_shallow_diffusion:
            self.train_aux_decoder = self.shallow_args['train_aux_decoder']
            self.train_diffusion = self.shallow_args['train_diffusion']
            self.aux_decoder_grad = self.shallow_args['aux_decoder_grad']
            self.aux_decoder = AuxDecoderAdaptor(
                in_dims=hparams['hidden_size'], out_dims=out_dims, num_feats=1,
                spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
                aux_decoder_arch=self.shallow_args['aux_decoder_arch'],
                aux_decoder_args=self.shallow_args['aux_decoder_args']
            )
        self.diffusion_type = hparams.get('diffusion_type', 'ddpm')
        self.backbone_type = compat.get_backbone_type(hparams)
        self.backbone_args = compat.get_backbone_args(hparams, self.backbone_type)
        if self.diffusion_type == 'ddpm':
            self.diffusion = GaussianDiffusion(
                out_dims=out_dims,
                num_feats=1,
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                backbone_type=self.backbone_type,
                backbone_args=self.backbone_args,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        elif self.diffusion_type == 'reflow':
            self.diffusion = RectifiedFlow(
                out_dims=out_dims,
                num_feats=1,
                t_start=hparams['T_start'],
                time_scale_factor=hparams['time_scale_factor'],
                backbone_type=self.backbone_type,
                backbone_args=self.backbone_args,
                spec_min=hparams['spec_min'],
                spec_max=hparams['spec_max']
            )
        else:
            raise NotImplementedError(self.diffusion_type)

    def forward(
            self, txt_tokens, mel2ph, f0, key_shift=None, speed=None,
            spk_embed_id=None, languages=None, gt_mel=None, infer=True, **kwargs
    ) -> ShallowDiffusionOutput:
        condition, tpse_pred, gst_pred = self.fs2(
            txt_tokens, mel2ph, f0, key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, languages=languages, mel=gt_mel, infer=infer, 
            **kwargs
        )
        if infer:
            if self.use_shallow_diffusion:
                aux_mel_pred = self.aux_decoder(condition, infer=True)
                aux_mel_pred *= ((mel2ph > 0).float()[:, :, None])
                if gt_mel is not None and self.shallow_args['val_gt_start']:
                    src_mel = gt_mel
                else:
                    src_mel = aux_mel_pred
            else:
                aux_mel_pred = src_mel = None
            mel_pred = self.diffusion(condition, src_spec=src_mel, infer=True)
            mel_pred *= ((mel2ph > 0).float()[:, :, None])
            return ShallowDiffusionOutput(aux_out=aux_mel_pred, diff_out=mel_pred, gst_output=gst_pred, tpse_output=tpse_pred)
        else:
            if self.use_shallow_diffusion:
                if self.train_aux_decoder:
                    aux_cond = condition * self.aux_decoder_grad + condition.detach() * (1 - self.aux_decoder_grad)
                    aux_out = self.aux_decoder(aux_cond, infer=False)
                else:
                    aux_out = None
                if self.train_diffusion:
                    diff_out = self.diffusion(condition, gt_spec=gt_mel, infer=False)
                else:
                    diff_out = None
                return ShallowDiffusionOutput(aux_out=aux_out, diff_out=diff_out, gst_output=gst_pred, tpse_output=tpse_pred)

            else:
                aux_out = None
                diff_out = self.diffusion(condition, gt_spec=gt_mel, infer=False)
                return ShallowDiffusionOutput(aux_out=aux_out, diff_out=diff_out, gst_output=gst_pred, tpse_output=tpse_pred)


class DiffSingerVariance(CategorizedModule, ParameterAdaptorModule):
    @property
    def category(self):
        return 'variance'

    def __init__(self, vocab_size):
        CategorizedModule.__init__(self)
        ParameterAdaptorModule.__init__(self)
        self.predict_dur = hparams['predict_dur']
        self.predict_pitch = hparams['predict_pitch']

        self.use_spk_id = hparams['use_spk_id']
        if self.use_spk_id:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.rr = RhythmRegulator()
        self.lr = LengthRegulator()
        self.diffusion_type = hparams.get('diffusion_type', 'ddpm')
        if self.predict_pitch:
            self.use_melody_encoder = hparams.get('use_melody_encoder', False)
            if self.use_melody_encoder:
                self.melody_encoder = MelodyEncoder(enc_hparams=hparams['melody_encoder_args'])
                self.delta_pitch_embed = Linear(1, hparams['hidden_size'])
            else:
                self.base_pitch_embed = Linear(1, hparams['hidden_size'])

            self.pitch_retake_embed = Embedding(2, hparams['hidden_size'])
            pitch_hparams = hparams['pitch_prediction_args']
            self.pitch_backbone_type = compat.get_backbone_type(hparams, nested_config=pitch_hparams)
            self.pitch_backbone_args = compat.get_backbone_args(pitch_hparams, backbone_type=self.pitch_backbone_type)
            if self.diffusion_type == 'ddpm':
                self.pitch_predictor = PitchDiffusion(
                    vmin=pitch_hparams['pitd_norm_min'],
                    vmax=pitch_hparams['pitd_norm_max'],
                    cmin=pitch_hparams['pitd_clip_min'],
                    cmax=pitch_hparams['pitd_clip_max'],
                    repeat_bins=pitch_hparams['repeat_bins'],
                    timesteps=hparams['timesteps'],
                    k_step=hparams['K_step'],
                    backbone_type=self.pitch_backbone_type,
                    backbone_args=self.pitch_backbone_args
                )
            elif self.diffusion_type == 'reflow':
                self.pitch_predictor = PitchRectifiedFlow(
                    vmin=pitch_hparams['pitd_norm_min'],
                    vmax=pitch_hparams['pitd_norm_max'],
                    cmin=pitch_hparams['pitd_clip_min'],
                    cmax=pitch_hparams['pitd_clip_max'],
                    repeat_bins=pitch_hparams['repeat_bins'],
                    time_scale_factor=hparams['time_scale_factor'],
                    backbone_type=self.pitch_backbone_type,
                    backbone_args=self.pitch_backbone_args
                )
            else:
                raise ValueError(f"Invalid diffusion type: {self.diffusion_type}")

        if self.predict_variances:
            self.pitch_embed = Linear(1, hparams['hidden_size'])
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_prediction_list
            })

            if self.diffusion_type == 'ddpm':
                self.variance_predictor = self.build_adaptor(cls=MultiVarianceDiffusion)
            elif self.diffusion_type == 'reflow':
                self.variance_predictor = self.build_adaptor(cls=MultiVarianceRectifiedFlow)
            else:
                raise NotImplementedError(self.diffusion_type)

        self.train_tpse = hparams['train_tpse']
        if self.train_tpse:
            tpse_hparams = hparams['tpse_args']
            self.gst = StyleEncoder(
                idim = hparams['audio_num_mel_bins'], 
                gst_tokens = tpse_hparams['gst_tokens'], 
                gst_token_dim = hparams['hidden_size'], 
                gst_heads = tpse_hparams['gst_heads'], 
                gru_layers = tpse_hparams['gst_gru_layers'],
                gru_units = tpse_hparams['gst_gru_hidden_size']
            )
            self.tpse = TPSE(
                output_size = hparams['hidden_size'], 
                n_layers = tpse_hparams['tpse_fc_layers'], 
                gru_layers = tpse_hparams['tpse_gru_layers'],
                gru_in_units = hparams['hidden_size'], 
                gru_units = tpse_hparams['tpse_gru_hidden_size']
            )
            self.train_me_tpse = tpse_hparams['train_me_tpse']
            if self.train_me_tpse and self.predict_pitch:
                if self.use_melody_encoder:
                    self.me_gst = StyleEncoder(
                        idim = hparams['audio_num_mel_bins'], 
                        gst_tokens = tpse_hparams['gst_tokens'], 
                        gst_token_dim = hparams['hidden_size'], 
                        gst_heads = tpse_hparams['gst_heads'], 
                        gru_layers = tpse_hparams['gst_gru_layers'],
                        gru_units = tpse_hparams['gst_gru_hidden_size']
                    )
                    self.me_tpse = TPSE(
                        output_size = hparams['hidden_size'], 
                        n_layers = tpse_hparams['tpse_fc_layers'], 
                        gru_layers = tpse_hparams['tpse_gru_layers'],
                        gru_in_units = hparams['hidden_size'], 
                        gru_units = tpse_hparams['tpse_gru_hidden_size']
                    )

    def forward(
            self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, mel2ph=None,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None, mel2note=None,
            base_pitch=None, pitch=None, pitch_expr=None, pitch_retake=None,
            variance_retake: Dict[str, Tensor] = None,
            spk_id=None, languages=None,
            infer=True, mel=None, **kwargs
    ):
        if self.use_spk_id:
            ph_spk_mix_embed = kwargs.get('ph_spk_mix_embed')
            spk_mix_embed = kwargs.get('spk_mix_embed')
            if ph_spk_mix_embed is not None and spk_mix_embed is not None:
                ph_spk_embed = ph_spk_mix_embed
                spk_embed = spk_mix_embed
            else:
                ph_spk_embed = spk_embed = self.spk_embed(spk_id)[:, None, :]  # [B,] => [B, T=1, H]
        else:
            ph_spk_embed = spk_embed = None

        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur,
            spk_embed=ph_spk_embed, languages=languages,
            infer=infer
        )

        if not self.predict_pitch and not self.predict_variances:
            return dur_pred_out, None, ({} if infer else None), None, None, None, None

        if mel2ph is None and word_dur is not None:  # inference from file
            dur_pred_align = self.rr(dur_pred_out, ph2word, word_dur)
            mel2ph = self.lr(dur_pred_align)
            mel2ph = F.pad(mel2ph, [0, base_pitch.shape[1] - mel2ph.shape[1]])

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)
        
        if self.train_tpse:
            if mel==None and infer==False:
                raise ValueError(f"Training GST requires mel input!")
            if mel is not None:
                gst_pred = self.gst(mel)
            else:
                gst_pred = None
            tpse_pred = self.tpse(condition.detach()) # grad stopping
            if not infer:
                condition = condition + gst_pred
            else:
                condition = condition + tpse_pred
        else:
            gst_pred = None
            tpse_pred = None

        if self.use_spk_id:
            condition += spk_embed

        if self.predict_pitch:
            if self.use_melody_encoder:
                melody_encoder_out = self.melody_encoder(
                    note_midi, note_rest, note_dur,
                    glide=note_glide
                )
                melody_encoder_out = F.pad(melody_encoder_out, [0, 0, 1, 0])
                mel2note_ = mel2note[..., None].repeat([1, 1, hparams['hidden_size']])
                melody_condition = torch.gather(melody_encoder_out, 1, mel2note_)
                if self.train_tpse and self.train_me_tpse:
                    if mel is not None:
                        me_gst_pred = self.me_gst(mel)
                    else:
                        raise ValueError(f"Training GST requires mel input!")
                    me_tpse_pred = self.me_tpse(melody_condition.detach()) # grad stopping
                    if not infer:
                        melody_condition = melody_condition + me_gst_pred
                    else:
                        melody_condition = melody_condition + me_tpse_pred
                else:
                    me_gst_pred = None
                    me_tpse_pred = None

                pitch_cond = condition + melody_condition
            else:
                pitch_cond = condition.clone()  # preserve the original tensor to avoid further inplace operations

            retake_unset = pitch_retake is None
            if retake_unset:
                pitch_retake = torch.ones_like(mel2ph, dtype=torch.bool)

            if pitch_expr is None:
                pitch_retake_embed = self.pitch_retake_embed(pitch_retake.long())
            else:
                retake_true_embed = self.pitch_retake_embed(
                    torch.ones(1, 1, dtype=torch.long, device=txt_tokens.device)
                )  # [B=1, T=1] => [B=1, T=1, H]
                retake_false_embed = self.pitch_retake_embed(
                    torch.zeros(1, 1, dtype=torch.long, device=txt_tokens.device)
                )  # [B=1, T=1] => [B=1, T=1, H]
                pitch_expr = (pitch_expr * pitch_retake)[:, :, None]  # [B, T, 1]
                pitch_retake_embed = pitch_expr * retake_true_embed + (1. - pitch_expr) * retake_false_embed

            pitch_cond += pitch_retake_embed
            if self.use_melody_encoder:
                if retake_unset:  # generate from scratch
                    delta_pitch_in = torch.zeros_like(base_pitch)
                else:
                    delta_pitch_in = (pitch - base_pitch) * ~pitch_retake
                pitch_cond += self.delta_pitch_embed(delta_pitch_in[:, :, None])
            else:
                if not retake_unset:  # retake
                    base_pitch = base_pitch * pitch_retake + pitch * ~pitch_retake
                pitch_cond += self.base_pitch_embed(base_pitch[:, :, None])

            if infer:
                pitch_pred_out = self.pitch_predictor(pitch_cond, infer=True)
            else:
                pitch_pred_out = self.pitch_predictor(pitch_cond, pitch - base_pitch, infer=False)
        else:
            pitch_pred_out = None

        if not self.predict_variances:
            return dur_pred_out, pitch_pred_out, ({} if infer else None), tpse_pred, gst_pred, me_tpse_pred, me_gst_pred

        if pitch is None:
            pitch = base_pitch + pitch_pred_out
        var_cond = condition + self.pitch_embed(pitch[:, :, None])

        variance_inputs = self.collect_variance_inputs(**kwargs)
        if variance_retake is not None:
            variance_embeds = [
                self.variance_embeds[v_name](v_input[:, :, None]) * ~variance_retake[v_name][:, :, None]
                for v_name, v_input in zip(self.variance_prediction_list, variance_inputs)
            ]
            var_cond += torch.stack(variance_embeds, dim=-1).sum(-1)

        variance_outputs = self.variance_predictor(var_cond, variance_inputs, infer=infer)

        if infer:
            variances_pred_out = self.collect_variance_outputs(variance_outputs)
        else:
            variances_pred_out = variance_outputs

        return dur_pred_out, pitch_pred_out, variances_pred_out, tpse_pred, gst_pred, me_tpse_pred, me_gst_pred
