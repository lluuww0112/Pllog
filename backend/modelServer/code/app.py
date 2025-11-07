from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

import jwt
from datetime import datetime, timedelta # 토큰 발급 및 만료 확인용
from zoneinfo import ZoneInfo
import os

from tables.utils import TryExcept, token_required
from tables.session import get_session
import interface

import numpy as np




# Flask App creation
app = Flask(__name__)
CORS(app, supports_credentials=True) # 쿠키를 프론트로 보낼 수 있도록 허용
SECRET_KEY=os.getenv("SIGNATURE_KEY")
app.config["SECRET_KEY"] = SECRET_KEY



# 분석 결과 반환
    # 감정 라벨값 반환
    # diary_encoder의 cls 벡터를 이용해 Musics 상의 top k index를 획득
@app.route("/getDiaryEncoder", methods=["POST"])
@TryExcept(message="가사 임베딩을 업로드하는 도중 오류가 발생했습니다")
def getRecommend():
    """
        data["diary_id"] : str
        data["text"] : list(float)
    """
    data = request.json
    text = data["text"]
    emotion = None

    # model inference
    emotion, norm_diary_cls = interface.model.get_diary_cls(text)
    norm_diary_cls = norm_diary_cls.detach().numpy() 

    # search index
    indices = interface.VectorBase.search_vector(norm_diary_cls)
    print(indices, flush=True)

    return jsonify({
        "status" : 1,
        "emotion" :  emotion[0],
        "indices" : indices.tolist()[0]
    })



# upload music ???
@app.route("/addVector", methods=["POST"])
@TryExcept(message="벡터 베이스에 음악을 추가하는 도중 오류가 발생했습니다")
def addVector():
    """
        data["vectors"] : 2차원 배열 (smaple_num, emb_dim)
    """
    data = request.json
    vectors = data["vectors"]

    vectors = np.array(vectors).astype("float32")
    interface.VectorBase.add_vector(vectors)

    return jsonify({
        "status" : 1
    })



if __name__=="__main__":    
    app.run(host="0.0.0.0", port=8080, debug=True)
    
