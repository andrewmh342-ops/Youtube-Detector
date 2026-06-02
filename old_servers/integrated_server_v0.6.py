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
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from typing import List

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
THRESHOLD = 0.05
image_min_ratio = 0.8
video_min_ratio = 0.8
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ================= 2. HybridDeepfakeScanner 클래스 =================
class HybridDeepfakeScanner:
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

        self.base_thresholds = {
            'sd1.5': 0.085,
            'sd2.1': 0.070,
            'sdxl': 0.050,
            #"vae_sd3": ("stabilityai/stable-diffusion-3-medium-diffusers", "vae") # 터미널에 huggingface-cli login 로 토큰 필요
        }

    def _prepare_models(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        models = {
            "vae_sd1.5": ("stabilityai/sd-vae-ft-mse", None),
            "vae_sd2.1": ("stabilityai/sd-vae-ft-ema", None),
            "vae_sdxl": ("stabilityai/sdxl-vae", None),
        }
        
        for folder, (hub_path, subfolder) in models.items():
            path = os.path.join(self.model_dir, folder)
            if not os.path.exists(path):
                print(f"다운로드 중: {folder}...")
                try:
                    if subfolder:
                        AutoencoderKL.from_pretrained(hub_path, subfolder=subfolder).save_pretrained(path)
                    else:
                        AutoencoderKL.from_pretrained(hub_path).save_pretrained(path)
                except Exception as e:
                    print(f"다운로드 실패: {e}")

        lpips_path = os.path.join(self.model_dir, "lpips_vgg.pth")
        if not os.path.exists(lpips_path):
            print("LPIPS 가중치 저장 중...")
            lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
            torch.save(lpips_model.state_dict(), lpips_path)

    def _load_models(self):
        print("모델 로딩 중...")
        try:
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd1.5")).to(self.device).eval()
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd2.1")).to(self.device).eval()
            self.vaes['sdxl'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sdxl")).to(self.device).eval()
            #self.vaes['sd3'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd3")).to(self.device).eval()

            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            self.lpips_loss.load_state_dict(torch.load(os.path.join(self.model_dir, "lpips_vgg.pth"), map_location=self.device))
            self.lpips_loss.eval()
            print("모든 VAE/LPIPS 모델 로드 완료.")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")

    def check_digital_traces(self, img_path):
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
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is None: return 0.5
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except Exception as e:
            print(f"복잡도 계산 실패: {e}")
            return 0.5

    def analyze_lota_bitplanes(self, img_path):
        """LOTA(ICCV 2025) 기반 최하위 3개 비트 플레인 노이즈 추출"""
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: return 0.0
            
            # 최하위 3개 비트 플레인 추출
            x0 = img & 1
            x1 = (img >> 1) & 1
            x2 = (img >> 2) & 1
            
            # 노이즈 맵 생성 (0~7 범위)
            z = (4 * x2) + (2 * x1) + x0
            
            # 신호 증폭 (스케일링: 0~255)
            z_scaled = (z * (255 / 7)).astype(np.uint8)
            gray_z = cv2.cvtColor(z_scaled, cv2.COLOR_BGR2GRAY)
            
            # MGPS(Maximum Gradient Patch Selection) 휴리스틱을 위한 그래디언트 연산
            grad_x = cv2.Sobel(gray_z, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_z, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            
            # 비정상적인 LSB 노이즈 패턴 척도 반환
            return float(np.mean(grad_mag))
        except:
            return 0.0

    def get_score(self, img_path):
        try:
            complexity = self.get_complexity(img_path)
            lota_noise_score = self.analyze_lota_bitplanes(img_path)
            
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

            base_thr = self.base_thresholds.get(detected_model, 0.055)
            adaptive_threshold = base_thr * (image_min_ratio + (1 - image_min_ratio) * complexity)

            # 하위 비트 노이즈 임계값
            lota_soft_threshold = 240.0  
            # AEROBLADE 오차가 임계값의 1.25배 이내로 낮아 가짜가 강하게 의심되는 경계선 구간
            suspicion_margin = 1.3      
            
            # 1. 기존 AEROBLADE 오차가 임계값보다 명확히 낮으면 '가짜' 확정
            if min_error < adaptive_threshold:
                final_verdict = "가짜"
            
            # 2. AEROBLADE는 간신히 통과(진짜로 오판)했지만, 의심스러운 점수대(margin 내부)이면서
            # 동시에 LSB 노이즈마저 가짜 특성(230점 미만)을 보이면 교차 검증으로 '가짜' 확정
            elif min_error < (adaptive_threshold * suspicion_margin) and lota_noise_score < lota_soft_threshold:
                final_verdict = "가짜"
                
            # 3. 위 조건에 해당하지 않으면 '진짜'로 판별
            else:
                final_verdict = "진짜"

            return min_error, complexity, lota_noise_score, adaptive_threshold, detected_model, final_verdict

        except Exception as e:
            print(f"분석 중 에러 발생: {e}")
            return 999.0, 0.5, 0.0, 0.055, "Unknown", "오류"

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
        # 기존 비디오 프로세싱 로직 유지 (NSG-VD 등 확장 가능)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return 999.0, 0.5, 0.055, "파일 오류", "None"

        dynamic_sample_rate = max(1, total_frames // target_samples)
        model_total_errors = {name: 0.0 for name in self.vaes.keys()}
        complexities = []
        extracted_count = 0
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame % dynamic_sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                complexities.append(np.clip(np.log1p(lap_var) / 7.0, 0, 1))
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    for name, vae in self.vaes.items():
                        recon = vae(img_tensor).sample
                        loss = self.lpips_loss(img_tensor, recon).item()
                        model_total_errors[name] += loss
                
                extracted_count += 1
                if extracted_count >= target_samples: break
            current_frame += 1
        cap.release()
        
        avg_errors = {name: total / extracted_count for name, total in model_total_errors.items()}
        detected_model = min(avg_errors, key=avg_errors.get)
        avg_score = avg_errors[detected_model]
        avg_complexity = sum(complexities) / len(complexities)

        base_thr = self.base_thresholds.get(detected_model, 0.055)
        adaptive_threshold = base_thr * (video_min_ratio + (1 - video_min_ratio) * avg_complexity)
        
        verdict = "가짜" if avg_score < adaptive_threshold else "진짜"
        return avg_score, avg_complexity, adaptive_threshold, verdict, detected_model

# ================= 3. FastAPI 서버 설정 =================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

scanner = HybridDeepfakeScanner(MODEL_DIR)

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
        detected_model = "N/A" # 초기값

        if file_ext in video_extensions:
            score, complexity, current_threshold, verdict, detected_model = scanner.process_video(file_path)
            method = "AEROBLADE (비디오 공간 분석)"
            lota_score = 0.0
        else:
            has_watermark = scanner.check_digital_traces(file_path)
            if has_watermark:
                verdict, score, complexity, current_threshold, method, detected_model, lota_score = "가짜", 0.0, 0.5, 0.0, "메타데이터 흔적", "Metadata", 0.0
            else:
                score, complexity, lota_score, current_threshold, detected_model, verdict = scanner.get_score(file_path)
                method = "AEROBLADE + LOTA 비트플레인 융합"

        print(f"[{file.filename}] \n 판별 결과: {verdict} 점수: {score:.5f}, 로타 점수: {lota_score:.3f}, 방법: {method}, \n 적용 임계값: {current_threshold:.5f}), 복잡도: {complexity:.3f}, 탐지된 모델: {detected_model}")
                
        results.append({
            "filename": file.filename,
            "result": verdict,
            "detected_source": detected_model,
            "score": round(score, 5),
            "lota_noise_score": round(lota_score, 3),
            "threshold_used": round(current_threshold, 5),
            "complexity": round(complexity, 3),
            "method": method
        })

    if results:
        df = pd.DataFrame(results)
        csv_path = "detection_history.csv"
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
            
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408) # 로컬 호스트에서 실행 host="127.0.0.1" | 외부 접속 허용 host="0.0.0.0"