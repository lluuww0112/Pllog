import os

import torch

from dataload import DiaryLyricData
from model import LyricEncder, DiaryEncoder

import pickle

# get saved model path
_base_dir = os.path.dirname(__file__)
_lyric_weight_path = os.path.join(_base_dir, "model_save", "Cross_lyricEncoder.pth")
_diary_weight_path = os.path.join(_base_dir, "model_save", "Cross_diaryEncoder.pth")

# load model
lyricEncoder = LyricEncder()
lyricEncoder = torch.load(_lyric_weight_path, weights_only=False)

diaryEncoder = DiaryEncoder()
diaryEncoder = torch.load(_diary_weight_path, weights_only=False)

# =========================================================================

# Lyric Encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
lyric_result = []
lyricEncoder.to(device)
lyricEncoder.eval()

for idx in range(len(DiaryLyricData)):
    lyric = DiaryLyricData[idx]["lyric"]
    with torch.no_grad():
        encoded = lyricEncoder.encode(lyric)
    encoded = encoded.to("cpu")
    lyric_result.append(encoded)


# save lyricVector
with open("./vec_index.pkl", "wb") as f:
    lyric_result_tensor = torch.cat(lyric_result, dim=0) 
    pickle.dump(lyric_result_tensor.tolist() , f)

print("lyric encoded vector 저장 완료", lyric_result_tensor.shape)

# =========================================================================

# Diary Encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
diary_result = []
diaryEncoder.to(device)
diaryEncoder.eval()


for idx in range(len(DiaryLyricData)):
    diary = DiaryLyricData[idx]["diary"]
    with torch.no_grad():
        _, encoded = diaryEncoder.encode(diary)

    encoded = encoded.to("cpu").unsqueeze(0)
    diary_result.append(encoded)

# save diaryVector
with open("./vec_index_diary.pkl", "wb") as f:
    diary_result_tensor = torch.cat(diary_result, dim=0) 
    pickle.dump(diary_result_tensor.tolist() , f)


print("diary encoded vector 저장 완료", diary_result_tensor.shape)

