import time
from typing import Any
import librosa
import torchaudio
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as L
import pytorch_lightning.loggers as Loggers
import pytorch_lightning.callbacks as Cb
from nvsr_unet import NVSR
import numpy as np
from dataset import DistanceDataModule, DAY_1_FOLDER, DAY_2_FOLDER
##### VOICEFIXER
from voicefixer import Vocoder
##### VOICEFIXER

class PredictionDataset(Dataset):
    def __init__(self):
       pass
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        X = 32768
        I, _ = torchaudio.load("/Users/mikesol/Downloads/414_near.wav", frame_offset=idx*X, num_frames=X)
        O, _ = torchaudio.load("/Users/mikesol/Downloads/414_far.wav", frame_offset=idx*X, num_frames=X)
        return I, O

data_loader = DataLoader(PredictionDataset(), batch_size=1)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    model = NVSR.load_from_checkpoint('model.ckpt', map_location=torch.device('cpu'), channels=1)
    model.vocoder = Vocoder(sample_rate=44100)
    trainer = L.Trainer(accelerator="cpu")
    prediction = trainer.predict(model, data_loader)
    # for x, pred in enumerate(prediction):
    #     torchaudio.save(f"./temp/prediction{x}.wav", torch.from_numpy(pred).unsqueeze(0), 44100)
    torchaudio.save(f"./temp/prediction.wav", torch.cat([torch.from_numpy(x).unsqueeze(0) for x in prediction],1), 44100)