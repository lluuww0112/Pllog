#!/bin/sh
set -e

echo ">>> waiting for DB"
./wait-for-it.sh db:3306 --timeout=600 -- python3 code/init_db.py || {
    echo "DB 초기화 중 오류 발생. 컨테이너를 종료합니다."
    exit 1
}
echo ">>> DB init completed"


echo ">>> Execute session server with gunicorn"
gunicorn -w 4 -b 0.0.0.0:8080 app:app --timeout 500