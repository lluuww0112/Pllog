from flask import request, jsonify, current_app
import jwt

from functools import wraps
import traceback
import os
import requests


SERVER_NAME = os.getenv("SERVER_NAME")


# 예외처리 및 세션 관리 데코레이터
def TryExcept(message=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                traceback.print_exc()

                return jsonify({
                    "error_at": f"Check server({SERVER_NAME})'s error",
                    "specific" : str(e),
                    "message": message,
                    "status" : 0
                }), 500
            finally:
                try:
                    pass
                except Exception as remove_error:
                    print("Session Remove Failed", remove_error, flush=True)
        return wrapper
    return decorator


# 세션 확인을 위한 데코레이터
def token_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = request.cookies.get('session_token')

        if not token:
            return jsonify({"message": "로그인 해주세요"}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data["userID"]
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "로그인이 만료되었습니다"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "유효하지 않은 인증입니다"}), 401

        return func(current_user, *args, **kwargs)
    return wrapper