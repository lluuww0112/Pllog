import os

import torch
from torch.utils.data import DataLoader, Dataset

from dataload import DiaryLyricData
from model import LyricEncder

import pickle
import warnings


# Load Model
_base_dir = os.path.dirname(__file__)
_lyric_weight_path = os.path.join(_base_dir, "model_save", "lyricEncoder.pth")

model = LyricEncder()
model = torch.load(_lyric_weight_path, weights_only=False)


# set Data
dataloader = DataLoader(
    dataset=DiaryLyricData,
    batch_size=64
)


# Lyric Encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
result = []
model.to(device)
model.eval()
for batch in dataloader:
    x = batch["lyric"]
    with torch.no_grad():
        encoded = model.encode(x)
    result.append(encoded)


# save lyricVector
with open("./vec_index.pkl", "wb") as f:
    pickle.dump(result, f)