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
train_diaries = train_data["diary"]
train_lyrics = train_data["lyric"]
train_emotions = train_data["emotion"]
DiaryLyricData = Diary_Lyric_dataset(train_diaries, train_lyrics, train_emotions)

test_data = pd.read_csv("./data/test.csv")
test_diaries = test_data["diary"]
test_lyrics = test_data["lyric"]
test_emotions = test_data["emotion"]
DiaryLyricData_Test = Diary_Lyric_dataset(test_diaries, test_lyrics, test_emotions)

