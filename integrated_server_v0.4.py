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

#pip install torch torchvision pandas numpy Pillow opencv-python diffusers lpips pillow-heif fastapi uvicorn python-multipart transformers accelerate huggingface_hub

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
THRESHOLD = 0.05  # AEROBLADE 판별 임계값
image_min_ratio = 0.8  #이미지 적응형 임계값 비율
video_min_ratio = 0.8  #영상 적응형 임계값 비율
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

        # 모델별 기본 임계값 설정 (SD3는 성능이 좋아 낮게 설정)
        self.base_thresholds = {
            'sd1.5': 0.055,
            'sd2.1': 0.050,
            'sdxl': 0.054,
            'sd3': 0.0125  # SD3를 위해 대폭 낮춘 임계값
        }

    def _prepare_models(self):
        """필요한 모델이 없으면 다운로드"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 모델 리스트 (이름, 허깅페이스 경로)
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
                try:
                    # subfolder 인자가 있을 경우 적용하여 다운로드
                    if subfolder:
                        AutoencoderKL.from_pretrained(hub_path, subfolder=subfolder).save_pretrained(path)
                    else:
                        AutoencoderKL.from_pretrained(hub_path).save_pretrained(path)
                except Exception as e:
                    print(f"❌ {folder} 다운로드 실패: {e}")

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
            #self.vaes['sd3'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd3")).to(self.device).eval()

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
        """이미지의 시각적 복잡도를 계산 (한글 경로 지원 버전)"""
        try:
            # cv2.imread 대신 numpy를 사용하여 한글 경로 지원
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is None: return 0.5
            
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except Exception as e:
            print(f"복잡도 계산 실패: {e}")
            return 0.5
        
    def get_brightness(self, img_path):
        """이미지의 평균 밝기 계산 (0~1)"""
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: return 0.5
            
            # 밝기 채널(Y) 추출을 위해 YCrCb 변환
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            brightness = np.mean(yuv[:, :, 0]) / 255.0
            return float(brightness)
        except:
            return 0.5

    def get_compression_level(self, img_path):
        """이미지의 압축 노이즈(블록 현상) 정도를 추정 (0~1)"""
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.0
            
            # 가로/세로 인접 픽셀 간의 차이(기울기)를 통해 블록 현상 추정
            # 압축이 심할수록 특정 8x8 경계에서 급격한 변화가 생김
            dx = np.diff(img, axis=1)
            dy = np.diff(img, axis=0)
            compression_score = (np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 255.0
            # 일반적인 이미지 범위에 맞게 클리핑 (대략적인 수치)
            return float(np.clip(compression_score * 10, 0, 1))
        except:
            return 0.0

    def get_score(self, img_path):
        try:
            complexity = self.get_complexity(img_path)
            
            # VAE 재구성 오차 계산
            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            error_dict = {}

            with torch.no_grad():
                for name, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    loss = self.lpips_loss(img_tensor, recon).item()
                    error_dict[name] = loss
            
            detected_model = min(error_dict, key=error_dict.get)
            min_error = error_dict[detected_model]

            # [핵심] 탐지된 모델에 따라 다른 베이스 임계값 적용
            base_thr = self.base_thresholds.get(detected_model, 0.055)
            
            # 적응형 임계값 계산 (해당 모델의 base_thr 기준)
            adaptive_threshold = base_thr * (image_min_ratio + (1 - image_min_ratio) * complexity)

            return min_error, complexity, adaptive_threshold, detected_model

        except Exception as e:
            print(f"❌ 분석 중 에러 발생 ({img_path}): {e}")
            return 999.0, 0.5, 0.055, "Unknown"
        
    def get_complexity_from_frame(self, frame_cv2):
        """프레임(Numpy Array)으로부터 직접 복잡도 계산"""
        try:
            gray = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except:
            return 0.5
        
    def get_integrated_threshold(self, base_thr, complexity, brightness, compression):
        """
        모든 환경 변수를 고려한 최종 임계값 산출
        """
        # 1. 기본 복잡도 적용
        ratio = image_min_ratio # 0.8
        thr = base_thr * (ratio + (1 - ratio) * complexity)
        
        # 2. 밝기 보정
        # 밝기가 극단적(0.1 미만 또는 0.9 초과)일 때 오차가 커지는 경향을 반영하여 임계값 하향
        if brightness < 0.2 or brightness > 0.8:
            thr *= 0.9  # 임계값을 10% 낮추어 더 엄격하게 판정
            
        # 3. 압축률 보정
        # 압축 노이즈가 심할수록 진짜 이미지도 오차가 커지므로 임계값을 약간 낮춤
        # 압축이 심하면(1에 가까우면) 최대 15%까지 임계값 하향
        thr *= (1.0 - (compression * 0.15))
        
        return float(thr)

    def process_video(self, video_path, target_samples=30):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0: return 999.0, 0.5, 0.5, 0.0, 0.055, "파일 오류", "None"

            dynamic_sample_rate = max(1, total_frames // target_samples)
            
            model_total_errors = {name: 0.0 for name in self.vaes.keys()}
            complexities, brightnesses, compressions = [], [], []
            extracted_count = 0
            current_frame = 0

            # 임시 파일 경로를 생성하여 프레임별 지표 계산 (한글 경로 대응을 위해 필요시)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if current_frame % dynamic_sample_rate == 0:
                    # 1. 지표 추출
                    complexities.append(self.get_complexity_from_frame(frame))
                    
                    # 프레임 배열에서 직접 밝기/압축률 계산 (함수 재활용을 위해 간단히 구현)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightnesses.append(np.mean(gray) / 255.0)
                    dx = np.diff(gray, axis=1)
                    dy = np.diff(gray, axis=0)
                    compressions.append(np.clip((np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 255.0 * 10, 0, 1))

                    # 2. VAE 점수 계산
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        for name, vae in self.vaes.items():
                            recon = vae(img_tensor).sample
                            model_total_errors[name] += self.lpips_loss(img_tensor, recon).item()
                    
                    extracted_count += 1
                    if extracted_count >= target_samples: break
                current_frame += 1
            cap.release()
            
            avg_errors = {name: total / extracted_count for name, total in model_total_errors.items()}
            detected_model = min(avg_errors, key=avg_errors.get)
            
            avg_score = avg_errors[detected_model]
            avg_complexity = sum(complexities) / len(complexities)
            avg_brightness = sum(brightnesses) / len(brightnesses)
            avg_compression = sum(compressions) / len(compressions)

            base_thr = self.base_thresholds.get(detected_model, 0.055)
            # [수정] 영상 분석에도 통합 임계값 함수 적용
            adaptive_threshold = self.get_integrated_threshold(base_thr, avg_complexity, avg_brightness, avg_compression)
            
            verdict = "가짜" if avg_score < adaptive_threshold else "진짜"
            return avg_score, avg_complexity, avg_brightness, avg_compression, adaptive_threshold, verdict, detected_model

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
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 변수 초기화
        score, complexity, brightness, compression, current_threshold = 0.0, 0.5, 0.5, 0.0, 0.055
        verdict, detected_model, method = "N/A", "N/A", "N/A"

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext in video_extensions:
            score, complexity, brightness, compression, current_threshold, verdict, detected_model = scanner.process_video(file_path)
            method = "AEROBLADE (영상 통합 분석)"
        else:
            if scanner.check_digital_traces(file_path):
                verdict, method, detected_model = "가짜", "메타데이터 흔적", "Metadata"
            else:
                score, complexity, _, detected_model = scanner.get_score(file_path)
                brightness = scanner.get_brightness(file_path)
                compression = scanner.get_compression_level(file_path)
                base_thr = scanner.base_thresholds.get(detected_model, 0.055)
                current_threshold = scanner.get_integrated_threshold(base_thr, complexity, brightness, compression)
                verdict = "가짜" if score < current_threshold else "진짜"
                method = "AEROBLADE (이미지 통합 분석)"

        # 결과 리스트 생성 (컬럼 추가)
        results.append({
            "filename": file.filename,
            "result": verdict,
            "detected_source": detected_model,
            "score": round(score, 5),
            "threshold_used": round(current_threshold, 5),
            "complexity": round(complexity, 3),
            "brightness": round(brightness, 3),   # 추가
            "compression": round(compression, 3), # 추가
            "method": method
        })

    if results:
        df = pd.DataFrame(results)
        csv_path = "detection_history.csv"
        
        # 파일이 없으면 새로 만들고(header 포함), 있으면 내용만 추가(mode='a')
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
            
    return results # 전체 결과 리스트 반환

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408) # 로컬 호스트에서 실행 host="127.0.0.1" | 외부 접속 허용 host="0.0.0.0"