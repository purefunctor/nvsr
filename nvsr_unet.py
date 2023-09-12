import sys

sys.path.append("/vol/research/dcase2022/sr_eval_vctk/testees")

from auraloss.freq import MultiResolutionSTFTLoss
import torch.utils
import librosa
import torch.nn as nn
import torch.utils.data
import os
import pytorch_lightning as pl
from fDomainHelper import FDomainHelper
from mel_scale import MelScale
import numpy as np
from torchlibrosa import STFT, ISTFT

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

EPS = 1e-9


def _find_cutoff(x, threshold=0.95):
    threshold = x[-1] * threshold
    for i in range(1, x.shape[0]):
        if x[-i] < threshold:
            return x.shape[0] - i
    return 0

def _get_cutoff_index(x):
    stft_x = np.abs(librosa.stft(x))
    energy = np.cumsum(np.sum(stft_x, axis=-1))
    return _find_cutoff(energy, 0.97)

def postprocessing(x, out):
    # Replace the low resolution part with the ground truth
    length = out.shape[0]
    cutoffratio = _get_cutoff_index(x)
    stft_gt = librosa.stft(x)
    stft_out = librosa.stft(out)
    stft_out[:cutoffratio, ...] = stft_gt[:cutoffratio, ...]
    out_renewed = librosa.istft(stft_out, length=length)
    return out_renewed

def _get_cutoff_index2(x):
    my_stft_real, mystft_imag = STFT()(x)
    my_stft_real, mystft_imag = ss(my_stft_real), ss(mystft_imag)
    stft_x = torch.clamp(my_stft_real**2 + mystft_imag**2, EPS, np.inf) ** 0.5
    stft_x = ss(stft_x).transpose(0,1)
    energy = torch.cumsum(torch.sum(stft_x, axis=-1), 0)
    return _find_cutoff(energy, 0.97)

def uu(x):
    return torch.unsqueeze(x, 0)

def ss(x):
    return torch.squeeze(x, 0)

def postprocessing2(x, out):
    # Replace the low resolution part with the ground truth
    x = uu(x)
    out = uu(out)
    length = out.shape[-1]
    cutoffratio = _get_cutoff_index2(x)
    stft_gt_real, stft_gt_imag = STFT()(x)
    stft_out_real, stft_out_imag = STFT()(out)
    stft_gt_real, stft_gt_imag, stft_out_real, stft_out_imag = ss(stft_gt_real), ss(stft_gt_imag), ss(stft_out_real), ss(stft_out_imag)
    stft_out_real[:cutoffratio, ...] = stft_gt_real[:cutoffratio, ...]
    stft_out_imag[:cutoffratio, ...] = stft_gt_imag[:cutoffratio, ...]
    stft_out_real, stft_out_imag = uu(stft_out_real), uu(stft_out_imag)
    out_renewed = ISTFT()(stft_out_real, stft_out_imag, length=length)
    out_renewed = ss(out_renewed)
    return out_renewed

def to_log(input):
    assert torch.sum(input < 0) == 0, (
        str(input) + " has negative values counts " + str(torch.sum(input < 0))
    )
    return torch.log10(torch.clip(input, min=1e-8))


def from_log(input):
    input = torch.clip(input, min=-np.inf, max=5)
    return 10**input


def trim_center(est, ref):
    diff = np.abs(est.shape[-1] - ref.shape[-1])
    if est.shape[-1] == ref.shape[-1]:
        return est, ref
    elif est.shape[-1] > ref.shape[-1]:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., int(diff // 2) : -int(diff // 2)], ref
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref
    else:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est, ref[..., int(diff // 2) : -int(diff // 2)]
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class BN_GRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer=1,
        bidirectional=False,
        batchnorm=True,
        dropout=0.0,
    ):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs):
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class NVSR(pl.LightningModule):
    def __init__(self, channels, vocoder=None):
        super(NVSR, self).__init__()

        model_name = "unet"

        self.channels = channels
        ##### VOICEFIXER
        self.vocoder = vocoder
        ##### VOICEFIXER

        self.downsample_ratio = 2**6  # This number equals 2^{#encoder_blcoks}

        self.loss = nn.L1Loss() # MultiResolutionSTFTLoss()

        self.f_helper = FDomainHelper(
            window_size=2048,
            hop_size=441,
            center=True,
            pad_mode="reflect",
            window="hann",
            freeze_parameters=True,
        )

        self.mel = MelScale(n_mels=128, sample_rate=44100, n_stft=2048 // 2 + 1)

        # masking
        self.generator = Generator(model_name)
        # print(get_n_params(self.vocoder))
        # print(get_n_params(self.generator))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def predict_step(self, predict_batch, batch_idx):
        x, _ = predict_batch
        _, mel = self.pre(x)
        out = self(mel)
        out = from_log(out["mel"])
        out = self.vocoder(out, cuda=False)
        out, _ = trim_center(out, x)
        out = out.numpy()
        out = np.squeeze(out)
        out = postprocessing(np.squeeze(x.numpy()), out)
        #out = out.to("cuda:0")
        #_, y = self.pre(y)
        #loss = self.loss(out, y)
        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        _, mel = self.pre(x)
        out = self(mel)
        out = from_log(out["mel"])
        #out = self.vocoder(out, cuda=False)
        out, _ = trim_center(out, x)
        #out = out.to("cuda:0")
        _, y = self.pre(y)
        loss = self.loss(out, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        _, mel = self.pre(x)
        out = self(mel)
        out = from_log(out["mel"])
        #out = self.vocoder(out, cuda=False)
        out, _ = trim_center(out, x)
        #out = out.to("cuda:0")
        _, y = self.pre(y)
        loss = self.loss(out, y)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    # def get_vocoder(self):
    #     return self.vocoder

    def get_f_helper(self):
        return self.f_helper

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return sp, mel_orig

    def forward(self, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(mel_orig)

def to_log(input):
    assert torch.sum(input < 0) == 0, (
        str(input) + " has negative values counts " + str(torch.sum(input < 0))
    )
    return torch.log10(torch.clip(input, min=1e-8))


def from_log(input):
    input = torch.clip(input, min=-np.inf, max=5)
    return 10**input


class BN_GRU(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer=1,
        bidirectional=False,
        batchnorm=True,
        dropout=0.0,
    ):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)

    def forward(self, inputs):
        # (batch, 1, seq, feature)
        if self.batchnorm:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)


class Generator(nn.Module):
    def __init__(self, model_name="unet"):
        super(Generator, self).__init__()
        if model_name == "unet":
            from components.unet import UNetResComplex_100Mb

            self.analysis_module = UNetResComplex_100Mb(channels=1)
        elif model_name == "unet_small":
            from components.unet_small import UNetResComplex_100Mb

            self.analysis_module = UNetResComplex_100Mb(channels=1)
        elif model_name == "bigru":
            n_mel = 128
            self.analysis_module = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Linear(n_mel, n_mel * 2),
                BN_GRU(
                    input_dim=n_mel * 2,
                    hidden_dim=n_mel * 2,
                    bidirectional=True,
                    layer=2,
                ),
                nn.ReLU(),
                nn.Linear(n_mel * 4, n_mel * 2),
                nn.ReLU(),
                nn.Linear(n_mel * 2, n_mel),
            )
        elif model_name == "dnn":
            n_mel = 128
            self.analysis_module = nn.Sequential(
                nn.Linear(n_mel, n_mel * 2),
                nn.ReLU(),
                nn.BatchNorm2d(1),
                nn.Linear(n_mel * 2, n_mel * 4),
                nn.ReLU(),
                nn.BatchNorm2d(1),
                nn.Linear(n_mel * 4, n_mel * 4),
                nn.ReLU(),
                nn.BatchNorm2d(1),
                nn.Linear(n_mel * 4, n_mel * 2),
                nn.ReLU(),
                nn.Linear(n_mel * 2, n_mel),
            )
        else:
            pass  # todo warning

    def forward(self, mel_orig):
        out = self.analysis_module(to_log(mel_orig))
        if type(out) == type({}):
            out = out["mel"]
        mel = out + to_log(mel_orig)
        return {"mel": mel}
