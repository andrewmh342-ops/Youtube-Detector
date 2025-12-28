from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# [보안 설정]
# 나중에 웹사이트나 확장 프로그램에서 이 서버로 접속할 수 있게 허용하는 설정입니다.
# 지금은 테스트이므로 모든 접속(*)을 허용합니다.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 받은 이미지를 저장할 폴더 만들기 (없으면 자동 생성)
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "서버가 정상적으로 켜져 있습니다!"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # 1. 이미지 저장 경로 설정
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    
    # 2. 서버 컴퓨터에 이미지 파일 저장
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # 3. 잘 받았다고 클라이언트에게 응답
    print(f"이미지 수신 성공: {file.filename}")
    return {"info": "이미지 전송 성공", "filename": file.filename}