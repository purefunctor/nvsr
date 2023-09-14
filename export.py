from pytorch_lightning import LightningModule
import numpy as np
from nvsr_unet import NVSR, from_log
import torch
import torchaudio
from torchlibrosa import STFT, ISTFT
from voicefixer import Vocoder

nvsr = NVSR.load_from_checkpoint(
    "checkpoint.ckpt", map_location=torch.device("cpu"), channels=1
)
nvsr.eval()


def uu(x):
    return torch.unsqueeze(x, 0)


def ss(x):
    return torch.squeeze(x, 0)


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


class NvsrLightning(LightningModule):
    def __init__(self):
        super().__init__()
        self.vocoder = Vocoder(sample_rate=44100)
        self.generator = nvsr.generator
        self.f_helper = nvsr.f_helper
        self.mel_scale = nvsr.mel
        self.stft = STFT()
        self.istft = ISTFT(onnx=True)

    def pre_process(self, x):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(x)
        mel_orig = self.mel_scale(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return sp, mel_orig

    def _find_cutoff(self, x, threshold=0.95):
        threshold = x[-1] * threshold
        for i in range(1, x.shape[0]):
            if x[-i] < threshold:
                return x.shape[0] - i
        return 0

    def _get_cutoff_index(self, x):
        EPS = 1e-9
        my_stft_real, mystft_imag = self.stft(x)
        my_stft_real, mystft_imag = ss(my_stft_real), ss(mystft_imag)
        stft_x = torch.clamp(my_stft_real**2 + mystft_imag**2, EPS, np.inf) ** 0.5
        stft_x = ss(stft_x).transpose(0, 1)
        energy = torch.cumsum(torch.sum(stft_x, axis=-1), 0)
        return self._find_cutoff(energy, 0.97)

    def post_process(self, x, out):
        # x = uu(x)
        # out = uu(out)
        length = out.shape[-1]
        cutoffratio = self._get_cutoff_index(x)
        stft_gt_real, stft_gt_imag = self.stft(x)
        stft_out_real, stft_out_imag = self.stft(out)
        stft_gt_real, stft_gt_imag, stft_out_real, stft_out_imag = (
            ss(stft_gt_real),
            ss(stft_gt_imag),
            ss(stft_out_real),
            ss(stft_out_imag),
        )
        stft_out_real[:cutoffratio, ...] = stft_gt_real[:cutoffratio, ...]
        stft_out_imag[:cutoffratio, ...] = stft_gt_imag[:cutoffratio, ...]
        stft_out_real, stft_out_imag = uu(stft_out_real), uu(stft_out_imag)
        out_renewed = self.istft(stft_out_real, stft_out_imag, length=length)
        out_renewed = ss(out_renewed)
        return out_renewed

    def forward(self, x_in):
        _, x = self.pre_process(x_in)
        x = self.generator(x)
        x = from_log(x["mel"])
        x = self.vocoder(x)
        x, _ = trim_center(x, x_in)
        x = x.squeeze(0)
        x_in = x_in.squeeze(0)
        x = self.post_process(x_in, x)
        return x


if __name__ == "__main__":
    model = NvsrLightning()

    sample, _ = torchaudio.load("./nt1_middle.wav", frame_offset=0, num_frames=44100 * 5)
    sample = sample.unsqueeze(0)

    model.to_onnx(
        "model.onnx",
        sample,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": [0, 2],  # batch size and sample count
            "output": [0, 2],  # batch size and sample count
        },
    )
