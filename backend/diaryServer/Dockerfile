
FROM server_base

COPY diaryServer/code ./code

# DB 종속성을 위한 스크립트. DB 초기화가 끝나기 전에는 서버를 실행하지 않도록 할 수 있음
RUN curl -o wait-for-it.sh https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh
RUN chmod +x wait-for-it.sh


# 엔트리 쉘 복사
COPY ./diaryServer/command.sh /command.sh
RUN chmod +x command.sh

