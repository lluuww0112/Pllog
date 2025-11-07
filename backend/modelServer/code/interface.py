# 인터페이스 정의 파일
# from sqlalchemy import select
# from sqlalchemy.orm import scoped_session, sessionmaker
# import pymysql

# from tables.session import Base, ENGINE, get_session
# from tables.tables import User, Diary
# from pymysql.err import IntegrityError

# import uuid
# from datetime import datetime, timedelta
# from zoneinfo import ZoneInfo
# import json
import numpy as np
import os
import pickle

import torch
from model import DiaryEncoder
from model import EMB_DIM

import faiss



# =============================== Model ===============================
# Model init
diary_encoder = DiaryEncoder()
# load model here
device = "cpu"

# 안전한 상대 경로 구성 (현재 파일 기준)
_base_dir = os.path.dirname(__file__)
_diary_weight_path = os.path.join(_base_dir, "diaryEncoder.pth")

# .pth 파라미터를 가중치로 로드
diary_encoder = DiaryEncoder()
diary_encoder = torch.load(_diary_weight_path, weights_only=False, map_location=device)
diary_encoder.device = device

# load parameter value to model here (.pth)
diary_encoder.eval()


# =============================== VectorBase ===============================
TOP_K = 5
# Vector Base definition (demo)
# you have to use real data for service
_base_dir = os.path.dirname(__file__)
_vec_base_path = os.path.join(_base_dir, "data", "vec_index.pkl")
with open(_vec_base_path, "rb") as f:
    vec_base = pickle.load(f)

vec_base = np.array([sample for batch in vec_base for sample in batch])


emb_dim = EMB_DIM
vector_path = os.path.join(os.path.dirname(__file__), "vector.index")
# add data to base
index = faiss.IndexFlatL2(EMB_DIM)
index.add(vec_base)

# Load pre-saved data --> in real on-service
# faiss.read_index(vector_path)

# searching top k sim vector_index
# query = np.random.randn((768,)).astype("float32")
# top_k = 5
# distance, indices = index.search(query, top_k)



class model:
    def get_diary_cls(text : str):
        try:
            print("일기장 데이터로 부터 모델 추론 중", flush=True)
            emotion, norm_diary_cls = diary_encoder.encode(text)
            print("추론 완료", flush=True)

            return emotion, norm_diary_cls
        except:
            raise Exception("error at the model.get_dairy_cls")


class VectorBase:
    def search_vector(vector : np.array, top_k : int = 5):
        vector = vector.reshape(1, -1) 
        _, indices = index.search(vector, top_k) # distances, indexes
        return indices + 1 # mysql의 auto increment는 1부터 시작이기 때문에 여기서 인덱스를 통일

    def add_vector(vectors : np.array):
        try:
            vectors = np.array(vectors).astype("float32")
            index.add(vectors)
        except:
            raise Exception("error at VectorBase.add_vector")

