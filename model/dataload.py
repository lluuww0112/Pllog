import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader


class Diary_Lyric_dataset(Dataset):
    def __init__(self, diary, lyric, sentiments):
        super().__init__()

        emotion_mapper = {
            "슬픔" : 0,
            "행복" : 1,
            "희망" : 2,
            "분노" : 3,
            "사랑" : 4
        }

        self.diary = diary
        self.lyric = lyric
        self.sentiment = [emotion_mapper[sentiment] for sentiment in sentiments]

    def __len__(self):
        return len(self.diary)
    
    def __getitem__(self, index):
        return {
            "diary" : self.diary[index],
            "sentiment" : self.sentiment[index],
            "lyric" : self.lyric[index]
        }
    

train_data = pd.read_csv("./data/train.csv")
diaries = train_data["diary"]
lyrics = train_data["lyric"]
emotions = train_data["emotion"]

DiaryLyricData = Diary_Lyric_dataset(diaries, lyrics, emotions)
