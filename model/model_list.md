### Device select
- mps를 사용하는 경우 DEVICE == "mps"
- cuda를 사용하는 경우 DEVICE == "cuda"


### sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- sentence 표현에 특화된 모델
- 다국어 사용이 가능하며 경량화 되어 있음
- emb_dim : 384, ffn_dim : 1534
```json
{
    "EMB_DIM" : 384,
    "FFN_DIM" : 1534,
    "SENTIMENT_CLASSES" : 5,

    "DIARY_SENTIMENT_ENCODER" : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "DIARY_ABSTRACT_ENCODER" : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "LYRIC_ENCODER" : "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM" : 6,
    "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM" : 6,
    "DROPOUT" : 0.1,

    "OPTIMIZER" : "adamw",
    "LEARNING_RATE" : 1e-3,
    "EPOCHS" : 20,
    "BATCHH_SIZE": 64,
    "WARMUP_STEPS_RATE" : 0.1,
    "DTYPE" : "bf16",
    "CL_TEMPERATURE" : 0.7,

    "DEVICE" : "cuda"
}
```


### kcELECTRA
```json
{
    "EMB_DIM" : 768,
    "FFN_DIM" : 3072,
    "SENTIMENT_CLASSES" : 5,

    "DIARY_SENTIMENT_ENCODER" : "beomi/KcELECTRA-base-v2022",
    "DIARY_ABSTRACT_ENCODER" : "beomi/KcELECTRA-base-v2022",
    "LYRIC_ENCODER" : "beomi/KcELECTRA-base-v2022",

    "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM" : 6,
    "SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM" : 6,
    "DROPOUT" : 0.1,

    "OPTIMIZER" : "adamw",
    "LEARNING_RATE" : 1e-4,
    "EPOCHS" : 100,
    "BATCHH_SIZE": 64,
    "WARMUP_STEPS_RATE" : 0.1,
    "DTYPE" : "bf16",
    "CL_TEMPERATURE" : 0.7,

    "DEVICE" : "cuda"
}
```