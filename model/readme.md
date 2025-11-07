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

## 1. model train

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

## 2. configure parameter
```json
{
  "EMB_DIM": 384,
  "FFN_DIM": 1534,
  "SENTIMENT_CLASSES": 5,

  "DIARY_SENTIMENT_ENCODER": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "DIARY_ABSTRACT_ENCODER": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "LYRIC_ENCODER": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

  "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM": 5,
  "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM": 6,
  "DROPOUT": 0.1,

  "OPTIMIZER": "adamw",
  "LEARNING_RATE": 1e-4,
  "EPOCHS": 100,
  "BATCHH_SIZE": 64,
  "WARMUP_STEPS_RATE": 0.1,
  "DTYPE": "bf16",
  "CL_TEMPERATURE": 0.7,

  "DEVICE" : "cuda"
}
```
