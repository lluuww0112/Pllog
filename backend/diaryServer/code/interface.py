# 인터페이스 정의 파일
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import scoped_session, sessionmaker
import pymysql

from tables.session import Base, ENGINE, get_session
from tables.tables import User, Diary, Music, Recommendation
from pymysql.err import IntegrityError

import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import os
import pandas as pd


def make_tables():
    try:
        Base.metadata.create_all(ENGINE)
        print(">>> DB tables Init completed", flush=True)
    except:
        raise Exception(">>> Error while Initializing DB tables")


def add_musics():
    try:
        # music_path = os.path.join(os.path.dirname(__file__), "music", "musics.json")
        # with open(music_path, "r") as file:
        #     musics = json.load(file)["musics"]

        music_path = os.path.join(os.path.dirname(__file__), "music", "train.csv")
        musics_df = pd.read_csv(music_path)
        artists = musics_df["artist"]
        titles = musics_df["title"]
        lyrics = musics_df["lyric"]

        musics = [{"artist" : artist, "title" : title, "lyrics" : lyric} for artist, title, lyric in zip(artists, titles, lyrics)]
        musics = [Music(**music) for music in musics]
        with get_session(commit=True, rollback=True) as session:
            query = (
                select(Music)
            )
            result = session.execute(query).scalars().first()
            if not result:
                for music in musics:
                    session.add(music)
    except:
        raise Exception(">>> Error while adding dumy musics")



# =========================================================================


class user:
    def add_user(data, session):
        """
            data["user_id"] : str
            data["user_psw"] : str
        """
        try:
            session.add(User(**{
                "user_id" : data["user_id"],
                "user_psw" : data["user_psw"]
            }))
        except IntegrityError as e:
            if e.args[0] == 1062:
                raise Exception("이미 존재하는 유저입니다")
            else:
                raise Exception("error at user.add_user")
        
    
    def get_user(data, session : scoped_session):
        """
            data["user_id"] : str
        """

        try:
            query = (
                select(User)
                .where(User.user_id == data["user_id"])
            )

            result = session.execute(query).scalars().one_or_none()
            return result
        except Exception as e:
            raise Exception("error at user.get_user")
        
    
    def check_psw(data):
        """
           data["user_psw"]  : str
           data["user_orm"] : orm  class User
        """
        status = None
        message = None
        
        try:
            if not data["user_orm"]:
                status = 0
                message = "존재하지 않는 유저입니다"
            elif data["user_orm"].user_psw != data["user_psw"]:
                status = 0
                message = "비밀번호가 틀렸습니다"
            elif data["user_orm"].user_psw == data["user_psw"]:
                status = 1
                message = "환영합니다"

            return status, message
        except:
            raise Exception("error at user.check_psw")


class diary:
    def add_diary(data, session : scoped_session):
        """
            data["user_id"] : sre
            data["title"] : str
            data["text"] : str
            data["emotion"] : str
        """
        
        try:
            if None in [data[key] for key in data]:
                raise Exception("필수 기재 항목 : 제목, 본문")

            data["diary_id"] = uuid.uuid4().hex
            data["upload_date"] = datetime.now(ZoneInfo("Asia/Seoul"))
            session.add(Diary(**{
                "diary_id" : data["diary_id"],
                "user_id" : data["user_id"],
                "upload_date" : data["upload_date"],
                "title" : data["title"],
                "text" : data["text"],
                "emotion" : data["emotion"]
            }))
        except:
            raise Exception("error at diary.add_diary")
    
    def add_recommendations(data, session : scoped_session):
        """
            data["diary_id"] : uuid hex64
            data["music_index"] : list of int            
        """
        try:
            diary_id = data["diary_id"]
            recommends = data["music_index"]

            musics_recommended =[Recommendation(**{"diary_id" : diary_id, "music_index" : index}) for index in recommends]
            for music in musics_recommended:
                session.add(music)
        
        except:
            raise Exception("error at the diary.add_recommendations")
    

    def get_diary(data, session : scoped_session):
        """
            data["user_id"] : str
            data["diary_id"] : str | None
        """ 
        try:
            result = None
            if not data["diary_id"]: # 전체 일기장 조회
                query = (
                    select(Diary)
                    .where(Diary.user_id == data["user_id"])
                    .order_by(Diary.upload_date)
                )
                result = session.execute(query).scalars().all()
            elif data["diary_id"]: # 단일 일기장 조회
                query = (
                    select(Diary)
                    .where((Diary.user_id == data["user_id"]) &
                             (Diary.diary_id == data["diary_id"]))
                )
                result = session.execute(query).scalars().one_or_none()

            return result
        except:
            raise Exception("error at the diary.get_diary")
        
    
    def get_recommend(data, session : scoped_session):
        """
            data["user_id"] : str
            data["diary_id"] : str
        """

        try:
            query = (
                select(Recommendation)
                .join(Recommendation.music)
                .where(Recommendation.diary_id == data["diary_id"])
                .options(joinedload(Recommendation.music))
            )
            results = session.execute(query).scalars().unique().all()
            results = [result.music for result in results]

            return results

        except:
            raise Exception("error at the diary.get_recommend")
