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
from dataset import AudioDataset


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
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        x, y = val_batch
        _, mel = self.pre(x)
        out = self(mel)
        mel2 = from_log(out["mel"])
        out = self.vocoder(mel2, cuda=False)
        out, _ = trim_center(out, x)
        loss = self.loss(out, y)
        self.log('training_loss', loss)
        return loss

class AudioDataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        data = AudioDataset()
        audio_train, audio_val = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
        self.audio_train = audio_train
        self.audio_val = audio_val
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.audio_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.audio_val, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    model = NVSRMike(1)
    trainer = L.Trainer(accelerator="cpu")
    dm = AudioDataModule(batch_size=32)
    trainer.fit(model, dm)