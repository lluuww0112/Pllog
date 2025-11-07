from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests

import jwt
from datetime import datetime, timedelta # 토큰 발급 및 만료 확인용
from zoneinfo import ZoneInfo
import os

from tables.utils import TryExcept, token_required
from tables.session import get_session

import interface


MODEL_SERVER = os.getenv("MODEL_SERVER")


app = Flask(__name__)
CORS(app, supports_credentials=True) # 쿠키를 프론트로 보낼 수 있도록 허용
SECRET_KEY=os.getenv("SIGNATURE_KEY")
app.config["SECRET_KEY"] = SECRET_KEY



# 유저 가입 
@app.route("/regist", methods=["POST"])
@TryExcept(message="유저가입을 진행하는 도중 오류가 발생했습니다")
def regist():
    """
        data["user_id"] : str
        data["user_psw"] : str
    """
    data = request.json
    
    with get_session(commit=True, rollback=True) as session:
        interface.user.add_user(data, session)
    
    return jsonify({
        "status" : 1,
        "message" : "환영합니다",
    })


# 로그인 
@app.route("/login", methods=["POST"])
@TryExcept(message="로그인을 하는 도중 오류가 발생했습니다")
def login():
    """
        data["user_id"] : str
        data["user_psw"] : str
    """
    
    data = request.json
    with get_session() as session:
        result = interface.user.get_user(data, session)
        data["user_orm"] = result
        status, message = interface.user.check_psw(data)

    if status:
        token_payload = {
            "userID" : data["user_id"],
            "exp" :  datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(hours=1)
        }
        token = jwt.encode(token_payload, app.config["SECRET_KEY"], algorithm="HS256")

        # 응답 객체 생성 및 set_cookie 헤더 설정
        resp = make_response({
            "status" : status,
            "message" : message
        })
        resp.set_cookie('session_token', token, httponly=True, samesite='Lax')
        return resp
    else:        
        return jsonify({
            "status" : status,
            "message" : message
        })


# 로그아웃
@app.route("/logout", methods=["GET"])
@TryExcept(message="로그아웃 도중 문제가 발생했습니다")
@token_required
def logout(current_user):
    resp = make_response(jsonify({
        "status" : 1,
        "message" : "로그아웃 되었습니다"
    }))
    resp.set_cookie("session_token", '', expires=0)
    print(f"LogOut : {current_user}", flush=True)
    return resp


# 일기 업로드 
@app.route("/upload", methods=["POST"])
@TryExcept(message="일기장을 업로드 하는 도중 오류가 발생했습니다")
@token_required
def upload(current_user):
    """
        data["title"] : str
        data["text"] : str
    """
    data = request.json

    # get emotion & music recommnedation from modelServer
    payload = { # make payload
        "text" : data["text"]
    }
    resp = requests.post(f"{MODEL_SERVER}/getDiaryEncoder", json=payload, timeout=500)
    result = resp.json()

    if result["status"] == 0:
        raise Exception("모델 서버로 부터 결과를 받아오는 도중 오류 발생")
    elif result["status"] == 1:
        # get emotion & recommendataion indexes
        emotion = result["emotion"]
        music_index = result["indices"]
    
    # # hard codded for test
    # emotion = "중립"
    # music_index = [2, 3]
    
    data["emotion"] = emotion 
    data["user_id"] = current_user
    data["music_index"] = music_index

    # db 초기화시 추천할 노래 리스트들이 여기 있음을 가정
    with get_session(commit=True, rollback=True) as session:
        interface.diary.add_diary(data, session) # diary_id will added here
        interface.diary.add_recommendations(data, session) # add recommendations 

    return jsonify({
        "status" : 1,
        "message" : "업로드 완료"
    })



# 일기 조회 
@app.route("/getDiary", methods=["GET"], defaults={"diary_id" : None})
@app.route("/getDiary/<string:diary_id>", methods=["GET"])
@TryExcept(message="일기장을 조회하는 도중 오류가 발생했습니다")
@token_required
def getDiary(current_user, diary_id):
    """
        data["user_id"] : str
        data["diary_id"] : str | None
    """
    data = {"user_id" : current_user, "diary_id" : diary_id}

    with get_session(commit=True, rollback=True) as session:
        result = interface.diary.get_diary(data, session)
        if not diary_id: # 여러개 조회시
            result = [orm.to_dict() for orm in result]
        else: # 단일 조회시
            result = result.to_dict()
            musics = interface.diary.get_recommend(data, session)
            musics = [music.to_dict() for music in musics]
            result["recommend"] = musics

    return jsonify({
        "status" : 1,
        "result" : result
    })



# 노래 좋아요 누르기 ???



# 내가 좋아요 누른 노래 조회 ???




if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
