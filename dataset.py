import torchaudio
from torch.utils.data import Dataset
from scipy.fft import rfft
import sys
from scipy.signal import lfilter, butter


class AudioDataset(Dataset):

    def __init__(self, file_index=None):
        self.use_single_file = file_index is not None
        I = '87_near.wav'
        O = '87_far.wav'
        WI = torchaudio.info(I)
        WO = torchaudio.info(O)
        print(WI,WO)
        self.min_l = min(WI.num_frames, WO.num_frames)
        self.CHUNK = 2048


    def __len__(self):
        return self.min_l // self.CHUNK

    def __getitem__(self, i):
        I = '87_near.wav'
        O = '87_far.wav'
        x, _ = torchaudio.load(I, frame_offset=i*self.CHUNK, num_frames=self.CHUNK)
        y, _ = torchaudio.load(O, frame_offset=i*self.CHUNK, num_frames=self.CHUNK)
        return x, y