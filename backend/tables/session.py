from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from contextlib import contextmanager
import os


db_user = os.getenv("DB_USER")
psw = os.getenv("MYSQL_ROOT_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

ENGINE = create_engine(f"mysql+pymysql://{db_user}:{psw}@{db_host}/{db_name}?charset=utf8mb4")
Base = declarative_base()
Base.metadata.bind = ENGINE


# 컨텍스트 매니저를 이용하여 에러 발생시 자동으로 close되도록 할 수 있음
@contextmanager
def get_session(commit=False, rollback=False):
    session = scoped_session(sessionmaker(bind=ENGINE))
    try:
        yield session
        if commit: 
            session.commit()
            print("Session Auto Comitted", flush=True)
    except Exception as e:
        if rollback: 
            session.rollback()
            print("Session Auto Rollbacked", flush=True)
        raise Exception(str(e))
    finally:
        session.remove()

