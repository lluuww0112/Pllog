# Project
``` shell
.
├── config.json # 모델 파라미터
├── dataload.py # 데이터셋 정의
├── model.py # 모델 정의
├── model_train.py # Train 객체 정의
└── model_save # 모델 저장 경로
    ├── diary_encoder.pth
    └── lyric_encoder.pth

```

# Model Arch
![Image](https://github.com/user-attachments/assets/1681c07a-c4cd-4e6d-989f-1d10e0f2250a)


# How to use

## 1. dataloader setting

```python
from dataload import Diary_Lyric_dataset
from torch.utils.data import Dataloader


diary # 일기 텍스트 원본 리스트
sentiment # 일기에 대한 감정 라벨
lyric # 가사 원본 리스트


# 데이터 셋 정의
dataset = Diary_Lyric_dataset(
    diary=diary,
    lyric=lyric,
    sentiment=sentiment
)

# 데이터 로더 정의
dataloader = Dataloader(
    dataset=dataset,
    batch_size=16,
    shuffle=True
)

```


## 2. trainning
```python

from model import DiaryEncoder, LyricEncoder
from model_train import Train

# 모델 불러오기
diary_encoder = DiaryEncoder()
lyric_encoder = LyricEncoder()

# trainner 선언
trainner = Train(
    diary_encoder = diary_encoder,
    lyric_encoder = lyric_encoder,
    dataloader = dataloader
)

# model train
trainner.full_train(
    diary_encoder_name="diary_encoder",
    lyric_encoder_name="lyric_encoder",
)

```

## 3. configure parameter
```json
{
    "EMB_DIM" : 768, // hidden 차원
    "FFN_DIM" : 3072, // transformer FFN 차원
    "SENTIMENT_CLASSES" : 5, // 감정 라벨 수

    "DIARY_SENTIMENT_ENCODER" : "BERT-base-multilingual-cased", // pretrained model for sentiment encoder
    "DIARY_ABSTRACT_ENCODER" : "BERT-base-multilingual-cased", // pretrained model for abstract encoder
    "LYRIC_ENCODER" : "BERT-base-multilingual-cased", // pretrained model for abstract encoder
    "TOKENIZER" : "BERT-base-multilingual-cased", // tokenizer 

    "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM" : 5, // latent encoder layer num
    "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM" : 6, // latent encoder header num
    "DROPOUT" : 0.1, // global dropout

    "OPTIMIZER" : "adam", // optimizer, choose one in adam, adamw, sgd, rmsprop
    "LEARNING_RATE" : 1e-5, // learning rate
    "EPOCHS" : 20, // epochs
    "WARMUP_STEPS_RATE" : 0.1, // warmup steps rate
    "DTYPE" : "bf16", // choose one in bf16, fp116, fp32
    "CL_TEMPERATURE" : 0.7 // contrastive temperature
}


```
