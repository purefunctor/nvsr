import time
from typing import Any
import librosa
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import os
from torch.utils.data import DataLoader
import auraloss
import sys
import pytorch_lightning as L
from nvsr_unet import NVSR
import numpy as np
from auraloss.freq import MultiResolutionSTFTLoss
from dataset import DistanceDataModule, DAY_1_FOLDER, DAY_2_FOLDER


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


class NVSRMike(NVSR):
    def __init__(self, channels):
        super(NVSRMike, self).__init__(channels)
        self.loss = MultiResolutionSTFTLoss()
        for l in self.loss.stft_losses:
            l.window = l.window.to("cuda:0")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        x, y = train_batch
        _, mel = self.pre(x)
        out = self(mel)
        mel2 = from_log(out["mel"])
        out = self.vocoder(mel2, cuda=False)
        out, _ = trim_center(out, x)
        loss = self.loss(out, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        x, y = val_batch
        _, mel = self.pre(x)
        out = self(mel)
        mel2 = from_log(out["mel"])
        out = self.vocoder(mel2, cuda=False)
        out, _ = trim_center(out, x)
        loss = self.loss(out, y)
        self.log("training_loss", loss)
        return loss


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model = NVSRMike(1)
    datamodule = DistanceDataModule(
        DAY_1_FOLDER, DAY_2_FOLDER, chunk_length=32768, num_workers=24
    )

    trainer = L.Trainer(max_epochs=20)
    trainer.fit(model, datamodule=datamodule)
