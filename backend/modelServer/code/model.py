import torch
import torch.nn as nn
import torch.quantization as quant

from transformers import AutoModel, AutoTokenizer
import os

import json
import time

config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as file:
    config = json.load(file)

EMB_DIM = config["EMB_DIM"]
FFN_DIM = config["FFN_DIM"]
SENTIMENT_CLASSES = config["SENTIMENT_CLASSES"]

DIARY_SENTIMENT_ENCODER = config["DIARY_SENTIMENT_ENCODER"]
DIARY_ABSTRACT_ENCODER = config["DIARY_ABSTRACT_ENCODER"]
LYRIC_ENCODER = config["LYRIC_ENCODER"]


SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM = config["SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM"]
SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM = config["SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM"]
DROPOUT = config["DROPOUT"]

DEVICE = config["DEVICE"]

class DiaryEncoder(nn.Module):

    def __init__(self,
                 diary_sentiment_encoder : str = DIARY_SENTIMENT_ENCODER,
                 diary_abstract_encoder : str = DIARY_ABSTRACT_ENCODER,
                 emb_dim : int = EMB_DIM,
                 ffn_dim : int = FFN_DIM,
                 latent_header_num : int = SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_HEADER_NUM,
                 latent_layer_num : int = SENTIMENT_ABSTRACT_LATENET_TRANSFORMER_LAYER_NUM,
                 dropout : float = DROPOUT,
                 classes : int = SENTIMENT_CLASSES,
                 gpu : str = DEVICE
                 ):
        super().__init__()
        
        # device
        self.device = self.select_device(gpu)
        # === tokenizer ===
        self.tokenizer = AutoTokenizer.from_pretrained(diary_abstract_encoder)
        
        # load pretrained model
        sentiment_pretrained = AutoModel.from_pretrained(diary_sentiment_encoder)
        abstract_pretrained = AutoModel.from_pretrained(diary_sentiment_encoder)
        
        # === emb ===
        self.sentiment_emb = sentiment_pretrained.embeddings
        self.abstract_emb = abstract_pretrained.embeddings
        
        # === sentiment & abstract encoder ===
        self.diary_sentiment_encoder = sentiment_pretrained.encoder
        self.diary_abstract_encoder = abstract_pretrained.encoder

        # sentiment header (for classify)
        self.diary_sentiment_header = nn.Sequential(*[
            nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True), # bert_pooler
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(in_features=emb_dim, out_features=classes, bias=True), # classify
            nn.Sigmoid() 
        ])


        # latent encoder (cross Attn)
        layers = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=latent_header_num,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.latent_encoder = nn.TransformerDecoder(
            decoder_layer=layers,
            num_layers=latent_layer_num
        )

        # latent header
        self.latent_header = nn.Sequential(*[
            nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True),
            nn.Tanh(), # 마지막 정규화 및 코사인 유사도를 기반으로 대조학습을 하기 위해 tanh을 추가
        ])

        self.idx2emotion = ["슬픔", "행복", "희망", "분노", "사랑"]

    
    def select_device(self, gpu : str):
        if gpu == "mps":
            return "mps" if torch.mps.is_available() else "cpu"
        elif gpu == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
    
    def forward(self, x : torch.Tensor, debug=False):
        """
            x : (batch, token_num) -> token_indx_sequence

            use [CLS] of sentiment_encoder for sentiment classfication & final Contrastive Learning
            so you have to set tokenizer to use special token
        """
        
        x = x.to(self.device)
        if not debug:            
            sentiment_input = self.sentiment_emb.forward(x)
            abstract_input = self.abstract_emb.forward(x[:, 1:]) # abstractor에는 cls토큰이 필요 없음
        else: # forwarding test 용
            sentiment_input = x
            abstract_input = x[:,1:,:]

        # sentiment forwarding
        sentiment_output = self.diary_sentiment_encoder.forward(sentiment_input)[0]
        sentiment_cls = sentiment_output[:, :1, :] # shape (batch, 1, emb_dim)
        sentiment_header_logit = self.diary_sentiment_header(sentiment_cls)

        # abstract forwarding
        abstract_output = self.diary_abstract_encoder.forward(abstract_input)[0] # shape (batch, seq_len, emb_dim)

        # latent forwarding
        final_output = self.latent_encoder(
            tgt=abstract_output, 
            memory=sentiment_cls,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        final_cls = self.latent_header(final_output[:,0,:])

        sentiment_logit = sentiment_header_logit.squeeze()
        diary_cls = final_cls.squeeze()

        return sentiment_logit, diary_cls
    
    
    def encode(self, x, debug=False):
        tokenized = self.tokenizer(
            x,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512
        )["input_ids"].to(self.device)
        
        logit, diary_cls = self.forward(tokenized, debug=debug)
        pred_label = torch.argmax(logit, dim=-1).long()
        pred_label = torch.atleast_1d(pred_label)
        
        # final processing 1. idx2emotion 2. make unit vector
        emotion = [self.idx2emotion[idx] for idx in pred_label.tolist()]
        norm_diary_cls = nn.functional.normalize(diary_cls, dim=-1, eps=1e-12) # normalize for vector searching
        return emotion, norm_diary_cls
        



class LyricEncder(nn.Module):
    def __init__(self, 
                 encoder : str = LYRIC_ENCODER,
                 emb_dim : int = EMB_DIM,
                ):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder)
        
        # lyric encoder
        pretrained = AutoModel.from_pretrained(encoder)
        self.emb_layer = pretrained.embeddings
        self.encoder = pretrained.encoder

        # header
        self.header = nn.Sequential(*[
            nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True),
            nn.Tanh(),
        ])


    def forward(self, x, debug=False):
        """
            x : (batch, token_num, emb_dim)

            to use [CLS], you have to set tokenizer to use special token
        """

        if not debug:
            embedded = self.emb_layer(x)
        else:
            embedded = x
        final_output = self.encoder(embedded)[0]
        lyric_cls = self.header(final_output[:, 0, :])  
        return lyric_cls

    
    def encode(self, x, debug=False):
        tokenized = self.tokenizer(
            x,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512
        )["input_ids"].to(self.device)

        lyric_cls = self.forward(tokenized, debug=debug)
        norm_lyric_cls = nn.functional.normalize(lyric_cls, dim=-1, eps=1e-12) # normalize for vector searching
        return norm_lyric_cls


# ============================================ test ============================================

if __name__ == "__main__":
    device = "cpu"
    print(device)
    model = DiaryEncoder().to(device)
    model.device = device
    
    start = time.time()
    emotion, final_cls = model.encode(["안녕", "나는 문어"])
    end = time.time()
    print(end - start)

    print(emotion)
    print(final_cls.shape)