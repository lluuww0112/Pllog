import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer

from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim import Adam, AdamW, SGD, RMSprop
from transformers import get_scheduler

from tqdm import tqdm
import json
import pandas as pd
import numpy as np

from model import DiaryEncoder, LyricEncder
from dataload import DiaryLyricData


with open("config.json", "r") as file:
    config = json.load(file)


OPTIMIZER = config["OPTIMIZER"]
LEARNING_RATE = config["LEARNING_RATE"]
EPOCHS = config["EPOCHS"]
BATCH_SIZE = config["BATCHH_SIZE"]
CL_TEMPERATURE = config["CL_TEMPERATURE"]
WARMUP_STEPS_RATE = config["WARMUP_STEPS_RATE"]

DIARY_TOKENIZER = config["DIARY_SENTIMENT_ENCODER"]
LYRIC_TOKENIZER = config["LYRIC_ENCODER"]

# Device selection: prioritize CUDA, then MPS, then CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


DTYPE = None
if config["DTYPE"] == "bf16":
   DTYPE = torch.bfloat16
elif config["DTYPE"] == "fp16":
   DTYPE = torch.float16
elif config["DTYPE"] == "fp32":
   DTYPE = torch.float32



class ContrastiveLoss(nn.Module):
    def __init__(self,
                 temperature : float = CL_TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / temperature))

    def forward(self,
                diary_cls : torch.Tensor,
                lyric_cls : torch.Tensor):
        """
            diary_cls : (batch, emb_dim)
            lyric_cls : (batch, emb_dim)
        """

        diary_cls_norm = F.normalize(diary_cls, dim=-1)
        lyric_cls_norm = F.normalize(lyric_cls, dim=-1)

        # calculate similarity
        batch_size = lyric_cls.shape[0]
        logit = self.logit_scale.exp() * diary_cls_norm @ lyric_cls_norm.T
        label = torch.arange(batch_size, device=lyric_cls.device)

        # calculate loss
        loss_diary = F.cross_entropy(logit, label)
        loss_lyric = F.cross_entropy(logit.T, label)
        contrasive_loss = (loss_diary + loss_lyric) / 2

        return contrasive_loss.mean()


class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ce_loss = nn.CrossEntropyLoss()
        self.cl_loss = ContrastiveLoss()

    def forward(self, sentiment_pred, sentiment_target, diary_cls, lyric_cls):
        cl_loss = self.cl_loss(diary_cls, lyric_cls)
        ce_loss = self.ce_loss(sentiment_pred, sentiment_target)
        
        return cl_loss + ce_loss


class Train:
    def __init__(self,
                 diary_encoder,
                 lyric_encoder,
                 dataset : Dataset,
                 diary_tokenizer : str = DIARY_TOKENIZER,
                 lyric_tokenizer : str = LYRIC_TOKENIZER,
                 optimizer : str = OPTIMIZER,
                 epochs : int  = EPOCHS,
                 batch_size : int = BATCH_SIZE,
                 lr : float = LEARNING_RATE,
                 device : str = DEVICE):
        
        # select device
        self.device = self.select_device(device)
        
        # model, tokenzier 
        self.diary_tokenizer = AutoTokenizer.from_pretrained(diary_tokenizer)
        self.lyric_tokenizer = AutoTokenizer.from_pretrained(lyric_tokenizer)

        self.diary_encoder = diary_encoder.to(self.device)
        self.lyric_encoder = lyric_encoder.to(self.device)

        self.dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size) # data
        self.epochs = epochs # epochs
        self.lr = lr # learning rate

        # optimizer & scaler
        self.optimizer = self.select_optimizer(optimizer_name=optimizer)
        # GradScaler는 CUDA일 때만 사용
        self.scaler = GradScaler('cuda') if self.device == "cuda" else None
        # loss function
        self.criterion = CustomLoss()

    
    def select_device(self, device : str):
        if device == "mps":
            return "mps" if torch.mps.is_available() else "cpu"
        elif device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
    
    def select_optimizer(self, optimizer_name: str):
        optimizer_name = optimizer_name.lower()
        params = list(self.diary_encoder.parameters()) + list(self.lyric_encoder.parameters())


        if optimizer_name == "adam":
            print(f"Optimizer: Adam, Learning Rate: {self.lr}")
            return Adam(params, lr=self.lr)
        
        elif optimizer_name == "adamw":
            print(f"Optimizer: AdamW, Learning Rate: {self.lr}")
            return AdamW(params, lr=self.lr)

        elif optimizer_name == "sgd":
            print(f"Optimizer: SGD, Learning Rate: {self.lr}")
            return SGD(params, lr=self.lr)
        
        elif optimizer_name == "rmsprop":
            print(f"Optimizer: RMSprop, Learning Rate: {self.lr}")
            return RMSprop(params, lr=self.lr)
        
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Please choose from 'adam', 'adamw', 'sgd'")
    

    def batch_train(self, diary, lyric, sentiment_target):
        """
            diary : (batch, token_num, emb_dim)
            lyric : (batch, token_num, emb_dim)
            sentiment_target : (batch, 1)

            diary & sentiment should be tokenized index sequence
        """
        # 모델이 실제로 있는 디바이스로 이동 (모델의 첫 번째 파라미터 디바이스 확인)
        model_device = next(self.diary_encoder.parameters()).device
        diary = diary.to(model_device).long()
        lyric = lyric.to(model_device).long()
        sentiment_target = sentiment_target.to(model_device).long()

        # autocast는 CUDA일 때만 사용, MPS와 CPU는 지원하지 않음
        if self.device == "cuda" and DTYPE is not None:
            with autocast(device_type='cuda', dtype=DTYPE):
                sentiment_label_predict, diary_cls = self.diary_encoder(diary)
                lyric_cls = self.lyric_encoder(lyric)

                loss = self.criterion.forward(
                    sentiment_pred=sentiment_label_predict, 
                    sentiment_target=sentiment_target, 
                    diary_cls=diary_cls, 
                    lyric_cls=lyric_cls
                )
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # CPU 또는 MPS는 일반적인 backward 사용
            sentiment_label_predict, diary_cls = self.diary_encoder(diary)
            lyric_cls = self.lyric_encoder(lyric)

            loss = self.criterion.forward(
                sentiment_pred=sentiment_label_predict, 
                sentiment_target=sentiment_target, 
                diary_cls=diary_cls, 
                lyric_cls=lyric_cls
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss
    
    def full_train(self,
                   diary_encoder_name : str,
                   lyric_encoder_name : str,
                   warmup_steps_rate : float = WARMUP_STEPS_RATE):
        """
            dataset : {
                "diary", --> low text
                "lyric", --> low text
                "sentiment" --> this is idx(long), shape (batch, 1)
            }
        """

        self.diary_encoder.to(self.device)
        self.lyric_encoder.to(self.device)
        self.diary_encoder.train()
        self.lyric_encoder.train()
        
        
        length = len(self.dataloader)

        num_training_steps = self.epochs * length
        warmup_steps = (int)(num_training_steps * warmup_steps_rate)

        # init scheduler
        scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        loss_history = []
        for epoch in range(self.epochs): 
            total_loss = 0
            
            tqdm_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in tqdm_bar:
                # tokenzing
                diary_tokenized = self.diary_tokenizer(
                    batch["diary"],
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=512  
                )
                diary_seq = diary_tokenized["input_ids"].to(self.device)
                
                lyric_tokenzied = self.lyric_tokenizer(
                    batch["lyric"],
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=512  
                )
                lyric_seq = lyric_tokenzied["input_ids"].to(self.device)

                sentiment_target = batch["sentiment"].to(self.device)
                
                batch_loss = self.batch_train(diary_seq, lyric_seq, sentiment_target)
                total_loss += batch_loss
                scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                tqdm_bar.set_postfix(loss=f"{batch_loss : .4f}", current_lr=f"{current_lr}")

            mean_loss = total_loss / length
            loss_history.append(mean_loss.item())
            print(f"Epoch {epoch + 1} / {self.epochs} Loss : {mean_loss}")


        print("Train Completed")
        torch.save(self.diary_encoder, f"./model_save/{diary_encoder_name}.pth")
        torch.save(self.lyric_encoder, f"./model_save/{lyric_encoder_name}.pth")
        print("Model saved")

        return loss_history
    


if __name__ == "__main__":
    diaryEncoder = DiaryEncoder()
    lyricEncoder = LyricEncder()

    trainner = Train(diaryEncoder, lyricEncoder, DiaryLyricData)
    loss_history = trainner.full_train(
        diary_encoder_name="diaryEncoder",
        lyric_encoder_name="lyricEncoder",
    )

    df = pd.DataFrame(np.array(loss_history))
    df.to_csv("loss_history.csv")