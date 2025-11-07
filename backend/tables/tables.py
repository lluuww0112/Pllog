# 테이블 객체 및 db세션 객체 선언
from sqlalchemy import Column, ForeignKey, ForeignKeyConstraint
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.dialects.mysql import TINYTEXT, MEDIUMTEXT
from sqlalchemy.orm import relationship

from datetime import datetime
from zoneinfo import ZoneInfo

from tables.session import Base


class extansoin:
    def to_dict(self, blacklist=None):
        temp = {}
        for column in self.__table__.columns:
            if not blacklist or column.name not in blacklist:
                temp[column.name] = getattr(self, column.name)
        return temp


class User(Base):
    __tablename__ = "Users"
    
    user_id = Column(String(50), primary_key=True)
    user_psw = Column(String(50), nullable=False)


class Music(Base, extansoin):
    __tablename__ = "Musics"

    music_index = Column(Integer, primary_key=True, autoincrement="auto")
    title = Column(String(100), nullable=False)
    artist = Column(String(100), nullable=False)
    lyrics = Column(MEDIUMTEXT, nullable=False)


class Diary(Base, extansoin):
    __tablename__ = "Diaries"
    
    diary_id = Column(String(50), primary_key=True)
    user_id = Column(
        String(50), 
        ForeignKey("Users.user_id", ondelete="cascade", onupdate="cascade"),
        nullable=False)
    upload_date = Column(DateTime, nullable=False)
    title = Column(String(100), nullable=False)
    text = Column(MEDIUMTEXT, nullable=False)
    emotion = Column(String(50), nullable=False)


class Recommendation(Base, extansoin):
    __tablename__ = "Recommendations"
    
    recom_index = Column(Integer, primary_key=True, autoincrement="auto")
    diary_id = Column(
        String(50),
        ForeignKey("Diaries.diary_id", ondelete="cascade", onupdate="cascade"),
        nullable=False
    )
    music_index = Column(
        Integer,
        ForeignKey("Musics.music_index", ondelete="cascade", onupdate="cascade"),
        nullable=False   
    )

    music = relationship(Music, uselist=False)


class Mylike(Base):
    __tablename__ = "Mylikes"
    
    likes_index = Column(Integer, primary_key=True, autoincrement="auto")
    user_id = Column(
        String(50),
        ForeignKey("Users.user_id", ondelete="cascade", onupdate="cascade"),
        nullable=False
    )
    music_index = Column(
        Integer,
        ForeignKey("Musics.music_index", ondelete="cascade", onupdate="cascade"),
        nullable=False
    )