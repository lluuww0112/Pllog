import os

import torch
from torch.utils.data import DataLoader

from dataload import DiaryLyricData
from model import LyricEncder

import pickle

# Load Model
_base_dir = os.path.dirname(__file__)
_lyric_weight_path = os.path.join(_base_dir, "model_save", "lyricEncoder.pth")

model = LyricEncder()
model = torch.load(_lyric_weight_path, weights_only=False)

# set Data
dataloader = DataLoader(
    dataset=DiaryLyricData,
    batch_size=64,
    shuffle=False,
)

# Lyric Encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
result = []
model.to(device)
model.eval()

for batch in dataloader:
    x = batch["lyric"]
    with torch.no_grad():
        encoded = model.encode(x)    # shape: (batch, emb_dim)
    
    encoded = encoded.to("cpu")      # GPU 텐서는 CPU로 이동
    result.append(encoded)

# (sample_num, emb_dim) 형태로 합치기
result_tensor = torch.cat(result, dim=0)  # ✅ 최종 2차원 텐서

# save lyricVector
with open("./vec_index.pkl", "wb") as f:
    pickle.dump(result_tensor.tolist() , f)

print("lyric encoded vector 저장 완료", result_tensor.tolist())
