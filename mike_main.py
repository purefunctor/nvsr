import time
from typing import Any
import librosa
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
import pytorch_lightning.loggers as Loggers
from nvsr_unet import NVSR
import numpy as np
from aurahack.freq import MultiResolutionSTFTLoss
from dataset import DistanceDataModule, DAY_1_FOLDER, DAY_2_FOLDER
logger = Loggers.WandbLogger(project="audio-nvsr")

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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model = NVSR(1)
    datamodule = DistanceDataModule(
        DAY_1_FOLDER, DAY_2_FOLDER, chunk_length=32768, num_workers=24
    )

    trainer = L.Trainer(max_epochs=20, logger=logger)
    trainer.fit(model, datamodule=datamodule)
