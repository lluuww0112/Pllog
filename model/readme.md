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
![Image](https://github.com/user-attachments/assets/f9941193-f234-48e8-9582-331ffc8527ff)


# How to use

## 1. dataloader setting

```python
from model_train import Train
from model import DiaryEncoder, LyricEncder
from dataload import DiaryLyricData
import matplotlib.pyplot as plt


diaryEncoder = DiaryEncoder()
lyricEncoder = LyricEncder()

trainner = Train(diaryEncoder, lyricEncoder, DiaryLyricData) # load trainner
loss_history = trainner.full_train(
    diary_encoder_name="diaryEncoder", # .pth file save name
    lyric_encoder_name="lyricEncoder", # .pth file save name
    warmup_steps_rate=0.1 # warmup_ratio
)

plt.plot(loss_history)
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
