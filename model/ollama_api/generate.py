import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import json



# 상대경로 사용을 위한 자식 디렉토리 추가
os.chdir("/content/drive/MyDrive/자연어처리실습/pllog")


# NGROK_URL = "https://semifused-alessandro-pecuniary.ngrok-free.dev"
NGROK_URL = "http://localhost:11434"


def get_prompt(lyric : str, emotion:str):
  return f"""
    아래는 노래 가사와 해당 노래에 대한 감정이다

    \"\"\"
    {lyric}
    \"\"\"
    {emotion}


    다음에 따라 ***한글만 사용해서*** 일기를 작성
    1) 1인칭 시점, 한국어 만으로, 가사의 내용을 직접적으로는 사용하지 않고
    2) 일기 형식을 갖추고 날짜 없이 내용만 작성
    3) 분량은 200~300자로
    4) 현재 감정에 집중해야함. 예를 들어 현재 슬픈 감정인데 희망찬 다짐들을 해서는 안됨
    """

def generate_diary(lyric : str, emotion:str):
  payload = {
    "model": "gemma3:27b",
    "prompt": get_prompt(lyric, emotion),
    "stream": False
  }

  res = requests.post(
      f"{NGROK_URL}/api/generate",
      data=json.dumps(payload),
      headers={"Content-Type": "application/json"}
  )

  return res.json()["response"]


data = pd.read_csv("./train.csv")
lyrics = data["lyrics"].to_list()
emotions = data["emotion"].to_list()

diaries = []
for i in tqdm(range(len(lyrics))):
  diaries.append(
      generate_diary(lyrics[i], emotions[i])
  )

data["diaries"] = diaries
data.to_csv("train.csv")