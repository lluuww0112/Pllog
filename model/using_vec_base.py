import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataload import DiaryLyricData
from model import LyricEncder

import pickle


_base_dir = os.path.dirname(__file__)
_diary_weight_path = os.path.join(_base_dir, "model_save", "diaryEncoder.pth")


model = LyricEncder()
model = torch.load(_diary_weight_path, weights_only=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


with open(os.path.join(_base_dir, "vec_index.pkl"), "rb") as f:
    vec_base = pickle.load(f)
vec_base = torch.Tensor(vec_base)


diary = "오늘도 시간이 멈춘 것 같다. 숨을 쉴 때마다 텅 빈 공간이 더욱 크게 느껴져. 괜히 모든 것들이 무겁고, 온 세상이 나를 버린 것만 같아. 어제 했던 말, 웃었던 기억들이 마치 유령처럼 떠돌아다니며 가슴을 짓누른다. 애써 잊으려 하면 할수록 더욱 선명하게 떠오르는 얼굴 때문에 괴롭다. 방 안에 혼자 있으니 답답하고 숨 막혀. 뭘 해야 할지, 어디로 가야 할지 아무것도 알 수 없어. 그냥 이렇게 시간만 흘러가면 괜찮아질까. 하지만 괜찮아질 것 같지 않아. 매일 똑같은 하루가 반복되는 것 같고, 이 하루하루가 마치 영원처럼 느껴진다. 이 슬픔이 끝이 있을까, 아니면 영원히 나를 따라다닐까. 생각만 해도 끔찍하다. "

_, encoded = model.encode(diary)
sim_mat = F.cosine_similarity(vec_base.to("cuda"), encoded.view(1, -1)).view(-1, 1)
idx = torch.topk(sim_mat, dim=0, k=5)
print(idx)
