import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from diffusers import AutoencoderKL
import lpips
from fastapi.responses import HTMLResponse
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from typing import List # 다중 파일 처리를 위해 추가

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
THRESHOLD = 0.07

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ================= 2. AerobladeScanner 클래스 (기존 로직 유지) =================
class AerobladeScanner:
    def __init__(self, model_dir, device=DEVICE):
        self.device = device
        self.vaes = {}
        self.model_dir = model_dir
        
        print(f"=== 시스템 정보: {self.device.upper()} 모드로 실행 중 ===")
        self._prepare_models()
        self._load_models()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def _prepare_models(self):
        """필요한 모델이 없으면 다운로드"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 모델 리스트 (이름, 허깅페이스 경로)
        models = {
            "vae_sd1.5": "stabilityai/sd-vae-ft-mse",
            "vae_sd2.1": "stabilityai/sd-vae-ft-ema",
            "vae_sdxl": "stabilityai/sdxl-vae"
        }
        
        for folder, hub_path in models.items():
            path = os.path.join(self.model_dir, folder)
            if not os.path.exists(path):
                print(f"다운로드 중: {folder}...")
                AutoencoderKL.from_pretrained(hub_path).save_pretrained(path)

        # LPIPS 가중치 확인
        lpips_path = os.path.join(self.model_dir, "lpips_vgg.pth")
        if not os.path.exists(lpips_path):
            print("LPIPS 가중치 저장 중...")
            lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
            torch.save(lpips_model.state_dict(), lpips_path)

    def _load_models(self):
        """모델을 VRAM/RAM에 로드"""
        print("모델 로딩 중...")
        try:
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd1.5")).to(self.device).eval()
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd2.1")).to(self.device).eval()
            self.vaes['sdxl'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sdxl")).to(self.device).eval()

            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            self.lpips_loss.load_state_dict(torch.load(os.path.join(self.model_dir, "lpips_vgg.pth"), map_location=self.device))
            self.lpips_loss.eval()
            print("✅ 모든 모델 로드 완료.")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")

    def check_digital_traces(self, img_path):
        """1차 검사: 메타데이터 AI 흔적 확인"""
        try:
            with Image.open(img_path) as img:
                if img.info:
                    keywords = ['parameters', 'prompt', 'negative prompt', 'steps:']
                    for key in img.info.keys():
                        if key.lower() in keywords:
                            return True
        except: pass
        return False

    def get_score(self, img_path):
        """2차 검사: AEROBLADE 재구성 오차 계산"""
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            errors = []
            with torch.no_grad():
                for _, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    loss = self.lpips_loss(img_tensor, recon).item()
                    errors.append(loss)
            return min(errors)
        except:
            return 999.0

# ================= 3. FastAPI 서버 설정 =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = AerobladeScanner(MODEL_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
# [수정된 /upload 엔드포인트 로직]
@app.post("/upload")
async def detect_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # 1. 원본 이미지 저장
        original_filename = f"orig_{file.filename}"
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. [수정] 한글 경로를 지원하는 이미지 로드 방식
        try:
            # np.fromfile을 사용하여 바이트로 읽은 후 cv2로 디코딩합니다.
            img_array = np.fromfile(original_path, np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"파일 로드 실패: {e}")
            continue

        if img_cv is None:
            print(f"이미지 디코딩 실패: {original_path}")
            continue
            
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 3. IF-ELSE 구조로 분석 경로 결정
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            pad_w, pad_h = int(w * 0.2), int(h * 0.2)
            y1, y2 = max(0, y - pad_h), min(img_cv.shape[0], y + h + pad_h)
            x1, x2 = max(0, x - pad_w), min(img_cv.shape[1], x + w + pad_w)
            
            face_crop = img_cv[y1:y2, x1:x2]
            crop_filename = f"inspect_face_{file.filename}"
            inspection_path = os.path.join(UPLOAD_DIR, crop_filename)
            
            # [수정] 한글 경로를 지원하는 이미지 저장 방식
            extension = os.path.splitext(crop_filename)[1] # .jpg, .png 등 추출
            result, encoded_img = cv2.imencode(extension, face_crop)
            if result:
                with open(inspection_path, mode='w+b') as f:
                    encoded_img.tofile(f)
            
            method_info = "AEROBLADE (얼굴 크롭 분석)"
        else:
            inspection_path = original_path
            method_info = "AEROBLADE (원본 전체 분석)"

        # 4. 분석 진행
        has_watermark = scanner.check_digital_traces(inspection_path)
        
        if has_watermark:
            verdict, score, method = "가짜", 0.0, "메타데이터 흔적 감지"
        else:
            score = scanner.get_score(inspection_path)
            verdict = "가짜" if score < THRESHOLD else "진짜"
            method = method_info

        results.append({
            "filename": file.filename,
            "result": verdict,
            "score": round(score, 5),
            "method": method
        })

    return results # 전체 결과 리스트 반환

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=20408)