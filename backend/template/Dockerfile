FROM python:3.11

LABEL maintainer="lluuww0112"
LABEL version="1.0.0"
LABEL description="플라스크 서버 베이스"


RUN apt-get update && \
    apt-get install -y curl 

COPY ./template/requirements.txt requirements.txt
RUN pip install -r requirements.txt


RUN mkdir code

# 공통 서버 코드 복사
RUN mkdir /code/tables
COPY tables ./code/tables


EXPOSE 8080