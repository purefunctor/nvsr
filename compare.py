from export import NvsrLightning
import numpy as np
import onnxruntime as ort
import torch
import torchaudio

sample, _ = torchaudio.load("./nt1_middle.wav", frame_offset=0, num_frames=44100 * 5)
sample = sample.unsqueeze(0)

# model = NvsrLightning()
# model.eval()
# lightning_prediction = model(sample).unsqueeze(0)
# torchaudio.save("./temp/lightning-prediction.wav", lightning_prediction, 44100)

ort_session = ort.InferenceSession("model.onnx")
outputs = ort_session.run(["output"], {"input": sample.numpy()})
torchaudio.save("./temp/onnx-prediction.wav", torch.tensor(outputs[0]).unsqueeze(0), 44100)
