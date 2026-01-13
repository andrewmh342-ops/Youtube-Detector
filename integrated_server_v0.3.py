import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
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

#pip install torch torchvision pandas numpy Pillow opencv-python diffusers lpips pillow-heif fastapi uvicorn python-multipart transformers accelerate

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
THRESHOLD = 0.055  # AEROBLADE 판별 임계값

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

    def get_complexity(self, img_path):
        """이미지의 시각적 복잡도를 계산 (0.0 ~ 1.0)"""
        try:
            # OpenCV를 이용해 그레이스케일로 로드
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.5
            
            # 라플라시안 분산으로 에지 강도 측정
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            
            # 로그 스케일 정규화 (보통 0~500 사이 값을 0~1로 매핑)
            # 수치 7.0은 데이터 특성에 따라 6.0~8.0 사이로 조정 가능합니다.
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except:
            return 0.5

    def get_score(self, img_path):
        """
        2차 검사: AEROBLADE 재구성 오차 및 적응형 임계값 계산
        반환값: (최저 오차 점수, 이미지 복잡도, 적용된 임계값)
        """
        try:
            # 1. 이미지 복잡도 계산
            complexity = self.get_complexity(img_path)
            
            # 2. 적응형 임계값 산출 (기본 0.055 기준)
            # 단순한 이미지(complexity 낮음)일수록 하한선(0.6)에 가까워져 임계값이 낮아짐
            base_threshold = 0.055
            min_ratio = 0.6
            adaptive_threshold = base_threshold * (min_ratio + (1 - min_ratio) * complexity)

            # 3. VAE 재구성 오차(LPIPS) 계산
            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            errors = []
            with torch.no_grad():
                for name, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    loss = self.lpips_loss(img_tensor, recon).item()
                    errors.append(loss)
            
            min_error = min(errors)
            
            # (점수, 복잡도, 임계값) 세 가지 정보를 모두 반환
            return min_error, complexity, adaptive_threshold

        except Exception as e:
            print(f"❌ 분석 중 에러 발생 ({img_path}): {e}")
            return 999.0, 0.5, 0.055
        
    def get_complexity_from_frame(self, frame_cv2):
        """프레임(Numpy Array)으로부터 직접 복잡도 계산"""
        try:
            gray = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except:
            return 0.5

    def process_video(self, video_path, target_samples=30):
        """
        영상을 샘플링하여 평균 점수와 평균 복잡도에 따른 판별 수행
        반환값: (평균 점수, 평균 복잡도, 적용된 임계값, 최종 판정)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return 999.0, 0.5, 0.055, "파일 오류"

        dynamic_sample_rate = max(1, total_frames // target_samples)
        
        scores = []
        complexities = []
        current_frame = 0
        extracted_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame % dynamic_sample_rate == 0:
                # 1. 복잡도 계산 (프레임에서 직접)
                complexities.append(self.get_complexity_from_frame(frame))
                
                # 2. AEROBLADE 점수 계산
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    errors = []
                    for name, vae in self.vaes.items():
                        recon = vae(img_tensor).sample
                        loss = self.lpips_loss(img_tensor, recon).item()
                        errors.append(loss)
                    scores.append(min(errors))
                
                extracted_count += 1
                if extracted_count >= target_samples:
                    break
            current_frame += 1
        
        cap.release()
        
        if not scores: return 999.0, 0.5, 0.055, "분석 불가"
        
        # 3. 평균치 계산 및 적응형 임계값 적용
        avg_score = sum(scores) / len(scores)
        avg_complexity = sum(complexities) / len(complexities)
        
        base_threshold = 0.055
        min_ratio = 0.6
        adaptive_threshold = base_threshold * (min_ratio + (1 - min_ratio) * avg_complexity)
        
        verdict = "가짜" if avg_score < adaptive_threshold else "진짜"
        
        return avg_score, avg_complexity, adaptive_threshold, verdict

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
async def detect_files(files: List[UploadFile] = File(...)):
    results = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 초기값 설정 (에러 방지 핵심)
        current_threshold = THRESHOLD
        complexity = 0.5
        method = "알 수 없음"

        if file_ext in video_extensions:
            # 영상 분석: 4개 값 수신
            score, complexity, current_threshold, verdict = scanner.process_video(file_path)
            method = "AEROBLADE (영상 분석)"
        else:
            # 이미지 분석
            has_watermark = scanner.check_digital_traces(file_path)
            if has_watermark:
                verdict, score, complexity, current_threshold, method = "가짜", 0.0, 0.5, 0.0, "메타데이터 흔적"
            else:
                # 이미지 분석: 3개 값 수신
                score, complexity, current_threshold = scanner.get_score(file_path)
                verdict = "가짜" if score < current_threshold else "진짜"
                method = "AEROBLADE (이미지 분석)"
        
        print(f"[{file.filename}] 판별 결과: {verdict} (점수: {score:.5f}, 방법: {method})")

        # 이제 모든 변수가 할당되었으므로 에러가 발생하지 않음
        results.append({
            "filename": file.filename,
            "result": verdict,
            "score": round(score, 5),
            "threshold_used": round(current_threshold, 5),
            "complexity": round(complexity, 3),
            "method": method
        })

    return results # 전체 결과 리스트 반환

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408) # 로컬 호스트에서 실행 host="127.0.0.1" | 외부 접속 허용 host="0.0.0.0"