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


diary = "오늘은 묘하게 마음이 간질거리는 날이다. 마치 오래 기다려온 선물을 받은 듯, 솜사탕처럼 달콤하고 부드러운 기분이 온몸을 감싼다. 평소에는 무심코 지나쳤던 작은 것들까지도 특별하게 느껴지고, 세상 모든 것이 장난처럼 느껴진다. 계속해서 누군가의 손길이 아른거리고, 보호해주고 싶은 마음이 간절하다. 그 사람을 향한 내 마음이 얼마나 큰지, 스스로도 놀랄 정도다. 가끔은 질투심도 들지만, 그마저도 사랑스러운 감정이다. 나는 지금 완전히 빠져버린 것 같다. 모든 것을 다 줄 수 있을 것 같고, 그 사람의 작은 손짓 하나에도 울컥하는 나를 발견한다. 이 감정이 얼마나 지속될지는 모르겠지만, 지금 이 순간만큼은 세상에서 가장 행복한 사람이다. 그냥 이대로 곁에 있고 싶다. 아무것도 바라지 않고, 그저 함께 시간을 보내고 싶다."
lyric = "오늘도 친구들이 왔어MAN HOW YOU BEEN WHATS UPAYE 여기 한 잔만 줄래제일 늦게 마시는 사람 술래그냥 섞어 마셔 CHAMPAGNEAND IF U KNOW WHAT I’M SAYIN내 손목을 보니 시간은 금이야불금이야 YOU DIG우린 젊기에 후회 따윈 내일 해조금 위험해AYE MAN YOU BETTER SLOW IT DOWN다시 돌아오지 않을 오늘을 위해저 하늘을 향해 건배해WE LIKE 2 PARTYYEAH YEAH YEAH YEAHWE LIKE 2 PARTY머리 위에 해 뜰 때까지WE LIKE 2 PARTYYEAH YEAH YEAH YEAHWE LIKE 2 PARTY서쪽에서 해 뜰 때까지오래간만에 불장난해지금 이 순간 나랑 같이 밖에 나갈래시끌 시끌 분위기는 환상겁이 없는 멋쟁이들 꽐라여기저기 널부러진 OPUS ONE에마무리는 달콤하게 D’yquem너는 빼지 않지 가지 함께 천국까지맨 정신은 반칙우린 젊기에 후회 따윈 내일 해조금 위험해AYE MAN YOU BETTER SLOW IT DOWN다시 돌아오지 않을 오늘을 위해저 하늘을 향해 건배해WE LIKE 2 PARTYYEAH YEAH YEAH YEAHWE LIKE 2 PARTY머리 위에 해 뜰 때까지WE LIKE 2 PARTYYEAH YEAH YEAH YEAHWE LIKE 2 PARTY서쪽에서 해 뜰 때까지너 없인 미쳐버리겠어DJ PLAY A LOVE SONG나 취한 게 아냐네가 보고 싶어 죽겠어SO DJ PLAY A LOVE SONGWE LIKE 2 PARTYYEAH YEAH YEAH YEAHWE LIKE 2 PARTY머리 위에 해 뜰 때까지WE LIKE 2 PARTYYEAH YEAH YEAH YEAHWE LIKE 2 PARTY서쪽에서 해 뜰 때까지"

emotion, norm_diary_cls = diaryEncoder.encode(diary)
norm_lyric_cls = lyricEncoder.encode(lyric)

vec_diary = norm_diary_cls.squeeze()
vec_lyric = norm_lyric_cls.squeeze()

# 2. 1D 벡터 간의 코사인 유사도를 계산합니다 (dim=0 사용).
cosine_similarity = F.cosine_similarity(vec_diary, vec_lyric, dim=0)

# 3. 결과 출력 (결과는 0차원 텐서이므로 .item()을 사용해 Python 숫자로 변환)
print("---" * 10)
print(f"일기-가사 코사인 유사도: {cosine_similarity.item()}")