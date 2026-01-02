import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import lpips
from pillow_heif import register_heif_opener
register_heif_opener()
from fastapi.responses import HTMLResponse
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from typing import List
import cv2

#pip install torch torchvision pandas numpy Pillow opencv-python diffusers lpips pillow-heif fastapi uvicorn python-multipart transformers accelerate

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
THRESHOLD = 0.065  # AEROBLADE 판별 임계값
THRESHOLD_FFT = 2.0  # 푸리에 변환 판별 임계값

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
# ================= 2. AerobladeScanner 클래스 =================
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

    def get_aero_score(self, img_path):
        """2차 검사: AEROBLADE 재구성 오차 계산"""
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            errors = []
            with torch.no_grad():
                for name, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    loss = self.lpips_loss(img_tensor, recon).item()
                    errors.append(loss)
            return min(errors)
        except Exception as e:
            # [수정] 터미널 창에 구체적으로 어떤 에러가 났는지 출력합니다.
            print(f"❌ 분석 중 에러 발생 ({img_path}): {e}")
            return 999.0
        
    def get_fft_score(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            
            img = cv2.resize(img, (512, 512))

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)

            rows, cols = img.shape
            crow, ccol = rows // 2 , cols // 2
            mask_size = 30
        
            magnitude_spectrum[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0

            score = np.mean(magnitude_spectrum)
            
            return float(score)
        except Exception as e:
            print(f"❌ 푸리에 분석 중 에러 발생: {e}")
            return 0.0

    def get_score(self, img_path):
        # 1. Aeroblade 점수 정규화 (가짜일 확률로 변환)
        # 임계값보다 낮을수록 1.0(가짜)에 가까워짐
        aero_score = self.get_aero_score(img_path)
        aero_prob = max(0, (THRESHOLD - aero_score) / THRESHOLD) if aero_score < THRESHOLD else 0

        # 2. 푸리에 점수 정규화 (가짜일 확률로 변환)
        # 임계값보다 높을수록 1.0(가짜)에 가까워짐
        fft_score = self.get_fft_score(img_path)
        fft_prob = min(1.0, fft_score / (THRESHOLD_FFT * 1.5)) # 1.5배 지점을 만점으로 가정

        # 3. 가중치 합산 (7:3 비율)
        final_fake_score = (aero_prob * 0.7) + (fft_prob * 0.3)

        # 4. 최종 판별
        if final_fake_score > 0.5:
            return 1, final_fake_score
        else:
            return 0, final_fake_score

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
    
@app.post("/upload")
async def detect_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # 1. 원본 이미지 저장
        original_filename = f"orig_{file.filename}"
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        inspection_path = original_path
        method_info = "원본 이미지 분석"

        # 4. 분석 진행
        has_watermark = scanner.check_digital_traces(inspection_path)
        
        if has_watermark:
            verdict, score, method = "가짜", 0.0, "메타데이터 흔적 감지"
        else:
            verdict,score = scanner.get_score(inspection_path)
            verdict = "가짜" if verdict == 1 else "진짜"
            method = method_info
        
        print(f"[{file.filename}] 판별 결과: {verdict} (점수: {score:.5f}, 방법: {method})")

        results.append({
            "filename": file.filename,
            "result": verdict,
            "score": round(score, 5),
            "method": method
        })

    return results # 전체 결과 리스트 반환

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408) # 로컬 호스트에서 실행 host="127.0.0.1" | 외부 접속 허용 host="0.0.0.0"
