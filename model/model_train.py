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
from dataload import DiaryLyricData, DiaryLyricData_Test



# 각 pair의 벡터가 얼마자 잘 정렬(가깝게)되어 있는가?
def align_loss(x, y, alpha=2): 
    """
        bsz : batch size (number of positive pairs)
        d   : latent dim
        x   : Tensor, shape=[bsz, d]
            latents for one side of positive pairs
        y   : Tensor, shape=[bsz, d]
            latents for the other side of positive pairs
    """
    return (x - y).norm(p=2, dim=1).pow(alpha).mean().to("cpu").item()


# L_uniform = uniform_loss(x) + uniform_loss(y)
# 두 인코더 각각의 출력 벡터 집합에 대해서 uniformnity를 측정후 더하면 됨
def uniform_loss(x, t=2): 
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log().to("cpu").item()


# 실제 추천결과 top5에 실제 정답이 몇 등에 위치하는지를 기반으로한 평가 지표
# topk = [1, 5, 10]에 대해서 본 함수는 구현되어 있음
# 실제 적용시에는 1, 5, 10에 대한 평균을 사용할 것
def calculate_recall(A: torch.Tensor, B: torch.Tensor, k_values: list = [1, 5, 10]):
    """
    두 벡터 집합 A, B 간의 Recall@K를 계산합니다.
    A[i]와 B[i]가 서로 대응되는 positive pair라고 가정합니다.

    Args:
        A (torch.Tensor): 쿼리 벡터 텐서 (N, embedding_dim)
        B (torch.Tensor): 후보 벡터 텐서 (N, embedding_dim)
        k_values (list): R@K를 계산할 K값들의 리스트

    Returns:
        dict: K값별 Recall 점수를 담은 리스트 (예: [0.5, 0.7, 0.8] 순서대로 R@1, R@5, R@10을 의미)
    """
    
    N = A.size(0)
    if N != B.size(0):
        raise ValueError("A와 B의 샘플 개수(N)가 일치해야 합니다.")

    # 1. 코사인 유사도 계산을 위해 벡터 정규화
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)

    
    sim_matrix = torch.matmul(A_norm, B_norm.T) # 2. (N, N) 크기의 전체 유사도 행렬 계산
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True) # 3. 각 쿼리(A[i])별로 후보(B)들의 유사도 순위 매기기
    ground_truth = torch.arange(N, device=A.device).view(-1, 1) # 4. 정답(Ground Truth) 생성
    ranks = (sorted_indices == ground_truth).nonzero(as_tuple=True)[1] # 5. 각 쿼리(A[i])에 대해 정답(B[i])이 몇 등인지(rank) 계산

    # 6. K 값들에 대해 Recall@K 계산
    recalls = {}
    for k in k_values:
        # 랭크가 k보다 작은 경우 (0-indexed이므로 R@1은 rank < 1)
        correct_at_k = (ranks < k).sum().item()
        
        # (정답 수 / 전체 쿼리 수)
        recalls[f"R@{k}"] = correct_at_k / N

    return [score for k, score in recalls.items()]


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
        recall_history = []
        alignment_history = []
        uniformnity_history = []

        val_recall_history = []
        val_alignment_history = []
        val_uniformnity_history = []

        for epoch in range(self.epochs): 
            total_loss = 0
            # train mode setting
            self.diary_encoder.train()
            self.lyric_encoder.train()

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

            # eval Mode Set (recall@K, alignment, uniformnity)
            self.diary_encoder.eval()
            self.lyric_encoder.eval()

            diaries = list(DiaryLyricData[:]["diary"])
            lyrics = list(DiaryLyricData[:]["lyric"])
            
            val_diaries = list(DiaryLyricData_Test[:]["diary"])
            val_lyrics = list(DiaryLyricData_Test[:]["lyric"])
            

            with torch.no_grad():
                # train metric
                _, diary_norm_vec = self.diary_encoder.encode(diaries)
                lyric_norm_vec = self.lyric_encoder.encode(lyrics)

                recalls = calculate_recall(diary_norm_vec, lyric_norm_vec)
                alignment = align_loss(diary_norm_vec, lyric_norm_vec)
                uniformnity = (uniform_loss(diary_norm_vec) + uniform_loss(lyric_norm_vec)) / 2

                
                # val metric
                _, diary_norm_vec = self.diary_encoder.encode(val_diaries)
                lyric_norm_vec = self.lyric_encoder.encode(val_lyrics)
                
                val_recalls = calculate_recall(diary_norm_vec, lyric_norm_vec)
                val_alignment = align_loss(diary_norm_vec, lyric_norm_vec)
                val_uniformnity = (uniform_loss(diary_norm_vec) + uniform_loss(lyric_norm_vec)) / 2


                
            recall_history.append(recalls)
            alignment_history.append(alignment)
            uniformnity_history.append(uniformnity)

            val_recall_history.append(val_recalls)
            val_alignment_history.append(val_alignment)
            val_uniformnity_history.append(val_uniformnity)

            mean_loss = total_loss / length
            loss_history.append(mean_loss.item())
            print(f"Epoch {epoch + 1} / {self.epochs} Loss : {mean_loss}")


        print("Train Completed")
        torch.save(self.diary_encoder, f"./model_save/{diary_encoder_name}.pth")
        torch.save(self.lyric_encoder, f"./model_save/{lyric_encoder_name}.pth")
        print("Model saved")

        return {
            "loss" : loss_history,
            "recall" : recall_history,
            "alignment" : alignment_history,
            "uniformnity" : uniformnity_history,
            "val_recall" : val_recall_history,
            "val_alignment" : val_alignment_history,
            "val_uniformnity" : val_uniformnity_history
        }


if __name__ == "__main__":
    diaryEncoder = DiaryEncoder()
    lyricEncoder = LyricEncder()

    mode = "Cross"
    diaryName = f"{mode}_diaryEncoder"
    lyricName = f"{mode}_lyricEncoder"

    trainner = Train(diaryEncoder, lyricEncoder, DiaryLyricData)
    history = trainner.full_train(
        diary_encoder_name=diaryName,
        lyric_encoder_name=lyricName,
    )

    loss_history = np.array(history["loss"]).reshape(-1, 1)
    recall_history = np.array(history["recall"]).reshape(-1, 3)
    alignment_history = np.array(history["alignment"]).reshape(-1, 1)
    uniformnity_history = np.array(history["uniformnity"]).reshape(-1, 1)

    full_history = pd.DataFrame(
        np.concat([loss_history, recall_history, alignment_history, uniformnity_history], axis=1),
        columns=["loss", "recall@1", "recall@5", "recall@10", "alignment", "uniformnity"]
    )
    full_history.to_csv(f"{mode}_history.csv", index=False)
