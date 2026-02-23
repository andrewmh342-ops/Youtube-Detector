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

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
IMAGE_MIN_RATIO = 0.8
VIDEO_MIN_RATIO = 0.8

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ================= 2. 통합 탐지 스캐너 클래스 =================
class IntegratedScanner:
    def __init__(self, model_dir, device=DEVICE):
        self.device = device
        self.vaes = {}
        self.model_dir = model_dir
        
        print(f"=== [시스템 정보] {self.device.upper()} 가동 시작 ===")
        self._prepare_models()
        self._load_models()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

        # AEROBLADE 베이스 임계값
        self.base_thresholds = {
            'sd1.5': 0.085,
            'sd2.1': 0.070,
            'sdxl': 0.050,
            'sd3': 0.0125
        }
        
        # FIRE 주파수 분석용 설정 (반지름 비율)
        self.r_low = 0.05
        self.r_high = 0.50

    def _prepare_models(self):
        """VAE 및 LPIPS 모델 준비"""
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        models = {
            "vae_sd1.5": ("stabilityai/sd-vae-ft-mse", None),
            "vae_sd2.1": ("stabilityai/sd-vae-ft-ema", None),
            "vae_sdxl": ("stabilityai/sdxl-vae", None),
            #"vae_sd3": ("stabilityai/stable-diffusion-3-medium-diffusers", "vae") # 터미널에 huggingface-cli login 로 토큰 필요
        }
        for folder, (hub_path, subfolder) in models.items():
            path = os.path.join(self.model_dir, folder)
            if not os.path.exists(path):
                print(f"다운로드 중: {folder}...")
                if subfolder:
                    AutoencoderKL.from_pretrained(hub_path, subfolder=subfolder).save_pretrained(path)
                else:
                    AutoencoderKL.from_pretrained(hub_path).save_pretrained(path)

        lpips_path = os.path.join(self.model_dir, "lpips_vgg.pth")
        if not os.path.exists(lpips_path):
            lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
            torch.save(lpips_model.state_dict(), lpips_path)

    def _load_models(self):
        """모델 메모리 로드"""
        try:
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd1.5")).to(self.device).eval()
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd2.1")).to(self.device).eval()
            self.vaes['sdxl'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sdxl")).to(self.device).eval()
            #self.vaes['sd3'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd3")).to(self.device).eval()

            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            self.lpips_loss.load_state_dict(torch.load(os.path.join(self.model_dir, "lpips_vgg.pth"), map_location=self.device))
            self.lpips_loss.eval()
            print("✅ 모든 포렌식 모델 로드 완료.")
        except Exception as e: print(f"❌ 로딩 실패: {e}")

    def check_digital_traces(self, img_path):
        """[복구] 1차 검사: 메타데이터 AI 흔적 확인"""
        try:
            with Image.open(img_path) as img:
                if img.info:
                    keywords = ['parameters', 'prompt', 'negative prompt', 'steps:']
                    for key in img.info.keys():
                        if key.lower() in keywords: return True
        except: pass
        return False
    
    def get_texture_crop(self, pil_img, crop_size=512):
        """
        이미지에서 텍스처(디테일)가 가장 풍부한 512x512 영역을 크롭함.
        """
        # 1. 계산을 위해 OpenCV 형식으로 변환 및 그레이스케일화
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 이미지 크기가 크롭 사이즈보다 작으면 패딩(여백) 추가
        if h < crop_size or w < crop_size:
            return pil_img.resize((crop_size, crop_size), Image.LANCZOS)

        # 2. 라플라시안 필터로 텍스처(에지) 강조
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)

        # 3. 512x512 윈도우로 합계가 가장 높은 지점 찾기 (효율성을 위해 보폭 설정 가능)
        # 여기서는 단순화를 위해 64픽셀 단위로 스캔
        best_score = -1
        best_coords = (0, 0)
        stride = 64 

        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                score = np.sum(laplacian[y:y+crop_size, x:x+crop_size])
                if score > best_score:
                    best_score = score
                    best_coords = (y, x)

        y, x = best_coords
        # 4. PIL 이미지 상태에서 크롭 (left, upper, right, lower)
        return pil_img.crop((x, y, x + crop_size, y + crop_size))

    def get_fire_score(self, img_tensor, recon_tensor):
        """FIRE 주파수 오차 점수 계산"""
        img_fft = torch.fft.fftshift(torch.fft.fftn(img_tensor, dim=(-2, -1)))
        recon_fft = torch.fft.fftshift(torch.fft.fftn(recon_tensor, dim=(-2, -1)))
        
        b, c, h, w = img_tensor.shape
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2).to(self.device)
        
        mask = (dist > (self.r_low * h)) & (dist < (self.r_high * h))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        diff_fft = torch.abs(img_fft - recon_fft)
        return ((diff_fft * mask).sum() / (mask.sum() + 1e-8)).item()

    def get_complexity(self, img_path):
        """이미지 시각적 복잡도 계산"""
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.5
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            return float(np.clip(np.log1p(laplacian_var) / 7.0, 0, 1))
        except Exception as e:
            print(f"복잡도 계산 실패: {e}")
            return 0.5

    def get_score(self, img_path_or_pil):
        """이미지/프레임 종합 점수 계산"""
        try:
            if isinstance(img_path_or_pil, str):
                image = Image.open(img_path_or_pil).convert("RGB")
                complexity = self.get_complexity(img_path_or_pil)
            else:
                image = img_path_or_pil
                complexity = 0.5

            image = self.get_texture_crop(image, crop_size=512)

            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            ae_dict, fire_dict = {}, {}

            with torch.no_grad():
                for name, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    ae_dict[name] = self.lpips_loss(img_tensor, recon).item()
                    fire_dict[name] = self.get_fire_score(img_tensor, recon)
            
            detected_model = min(ae_dict, key=ae_dict.get)
            min_ae_score = ae_dict[detected_model]
            min_fire_score = fire_dict[detected_model]
            
            base_thr = self.base_thresholds.get(detected_model, 0.055)
            adaptive_thr = base_thr * (IMAGE_MIN_RATIO + (1 - IMAGE_MIN_RATIO) * complexity)

            return min_ae_score, min_fire_score, complexity, adaptive_thr, detected_model
        except Exception as e:
            return 999.0, 999.0, 0.5, 0.055, "Error"

    def process_video(self, video_path, target_samples=30):
        """[복구] 비디오 시공간 분석 로직"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return 999.0, 999.0, 0.5, 0.055, "파일 오류", "None"

        sample_rate = max(1, total_frames // target_samples)
        ae_scores, fire_scores = [], []
        extracted_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or extracted_count >= target_samples: break
            
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                ae, fire, _, _, _ = self.get_score(pil_img)
                ae_scores.append(ae)
                fire_scores.append(fire)
                extracted_count += 1
        cap.release()
        
        avg_ae = sum(ae_scores) / len(ae_scores)
        avg_fire = sum(fire_scores) / len(fire_scores)
        
        # 비디오용 판별
        verdict = "가짜" if (avg_ae < 0.05) or (avg_fire < 0.15) else "진짜"
        return avg_ae, avg_fire, 0.5, 0.05, verdict, "Video_Model"

# ================= 3. FastAPI 서버 및 엔드포인트 =================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
scanner = IntegratedScanner(MODEL_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def detect_files(files: List[UploadFile] = File(...)):
    results = []
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

        if ext in video_exts:
            ae, fire, comp, thr, verdict, model = scanner.process_video(path)
            method = "AEROBLADE + FIRE (Video)"
        else:
            # 이미지 1차 메타데이터 검사
            if scanner.check_digital_traces(path):
                verdict, ae, fire, comp, thr, method, model = "가짜", 0.0, 0.0, 0.5, 0.0, "메타데이터", "Metadata"
            else:
                ae, fire, comp, thr, model = scanner.get_score(path)
                # 하이브리드 판별 (AE 임계값 미만 OR FIRE 점수 임계값 미만)
                verdict = "가짜" if (ae < thr*0.9) or (fire < 45.0) else "진짜"
                method = "AEROBLADE + FIRE (Image)"

        print(f"[{file.filename}] \n 판별 결과: {verdict} 점수: {ae:.5f}, 방법: {method}, \n 적용 임계값: {thr:.5f}), 복잡도: {comp:.3f}, 탐지된 모델: {model}")

        res = {
            "filename": file.filename,
            "result": verdict,
            "detected_source": model,
            "ae_score": round(ae, 5),
            "fire_score": round(fire, 5),
            "threshold": round(thr, 5),
            "complexity": round(comp, 3),
            "method": method
        }
        results.append(res)

    # CSV 로깅
    if results:
        df = pd.DataFrame(results)
        csv_path = "detection_history.csv"
        
        # 파일이 없으면 새로 만들고(header 포함), 있으면 내용만 추가(mode='a')
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
    
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408) # 로컬 호스트에서 실행 host="127.0.0.1" | 외부 접속 허용 host="