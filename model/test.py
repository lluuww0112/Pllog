import os
import torch
import torch.nn.functional as F
from model import DiaryEncoder, LyricEncder

diaryEncoder = DiaryEncoder()
lyricEncoder = LyricEncder()

_base_dir = os.path.dirname(__file__)
_diary_weight_path = os.path.join(_base_dir, "model_save", "diaryEncoder.pth")
_lyric_weight_path = os.path.join(_base_dir, "model_save", "lyricEncoder.pth")

diaryEncoder = torch.load(_diary_weight_path, map_location="cpu", weights_only=False)
lyricEncoder = torch.load(_lyric_weight_path, map_location="cpu", weights_only=False)


diary = "오늘도 시간이 멈춘 것 같다. 숨을 쉴 때마다 텅 빈 공간이 더욱 크게 느껴져. 괜히 모든 것들이 무겁고, 온 세상이 나를 버린 것만 같아. 어제 했던 말, 웃었던 기억들이 마치 유령처럼 떠돌아다니며 가슴을 짓누른다. 애써 잊으려 하면 할수록 더욱 선명하게 떠오르는 얼굴 때문에 괴롭다. 방 안에 혼자 있으니 답답하고 숨 막혀. 뭘 해야 할지, 어디로 가야 할지 아무것도 알 수 없어. 그냥 이렇게 시간만 흘러가면 괜찮아질까. 하지만 괜찮아질 것 같지 않아. 매일 똑같은 하루가 반복되는 것 같고, 이 하루하루가 마치 영원처럼 느껴진다. 이 슬픔이 끝이 있을까, 아니면 영원히 나를 따라다닐까. 생각만 해도 끔찍하다. "
lyric = "거짓말 날 위해 하는 말혼잣말 버릇이 된 이말괜찮아 질 거야 누구나겪는거야이런 하루가 반복되고하루하루 일년인 것 같아너 없는 오늘이하루하루 힘들 것만 같아숨쉬는 것 조차너의 흔적들이 아직 남아혼자 너무나도 아파I can't let it go마음은 움직일 수있어이별도 되돌릴 수있어기다려 줄게 나 이해해 줄게 다널 찾는 하루 반복되고하루하루 일년인 것 같아너 없는 오늘이하루하루 힘들 것만 같아숨쉬는 것 조차너의 흔적들이 아직 남아혼자 너무나도 아파I can't let it go하루하루없던 일처럼 돌릴 수 없니텅 빈 내방을 채울 수 없이흘러버린 추억들은잡을 수 없어하루하루 아플 것만 같아하루하루 죽을 만큼 아파너 없는 오늘이하루하루 멈출 것만 같아혼자인 세상이우리 추억들이 아직 남아가슴 깊이 파고 들어I can't let it go라라라라 I'll let u go라라라라 I'll let u go라라라라 하루하루라라라라 하루하루"

emotion, norm_diary_cls = diaryEncoder.encode(diary)
norm_lyric_cls = lyricEncoder.encode(lyric)

vec_diary = norm_diary_cls.squeeze()
vec_lyric = norm_lyric_cls.squeeze()

# 2. 1D 벡터 간의 코사인 유사도를 계산합니다 (dim=0 사용).
cosine_similarity = F.cosine_similarity(vec_diary, vec_lyric, dim=0)

# 3. 결과 출력 (결과는 0차원 텐서이므로 .item()을 사용해 Python 숫자로 변환)
print("---" * 10)
print(f"일기-가사 코사인 유사도: {cosine_similarity.item()}")