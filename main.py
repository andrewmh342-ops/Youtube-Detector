# main.py

from fastapi import FastAPI

# 1. FastAPI라는 간판을 단 식당(앱)을 만듭니다.
app = FastAPI()

# 2. 경로(Path) 만들기: 누군가 내 주소로 접속하면 이 함수를 실행해라!
# @app.get("/") -> "http://내주소/" 로 들어왔을 때
@app.get("/")
def read_root():
    # 3. 결과 돌려주기 (JSON 형식)
    return {"message": "반갑습니다! 여기가 바로 탐지 서버입니다."}

# 4. 가짜 의사 탐지용 경로 예시
@app.get("/check")
def check_fake():
    return {"status": "Deepfake Detection Server Running..."}