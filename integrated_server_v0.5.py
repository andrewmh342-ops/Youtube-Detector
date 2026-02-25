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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import time
import base64 
from typing import List

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"

THRESHOLD = 0.018  
image_min_ratio = 0.95  
video_min_ratio = 0.7  

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
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        models = {
            "vae_sd1.5": ("stabilityai/sd-vae-ft-mse", None),
            "vae_sd2.1": ("stabilityai/sd-vae-ft-ema", None),
            "vae_sdxl": ("stabilityai/sdxl-vae", None),
            "vae_sd3": ("stabilityai/stable-diffusion-3-medium-diffusers", "vae")
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
                    print(f"❌ {folder} 다운로드 실패: {e}")

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
            self.vaes['sd3'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd3")).to(self.device).eval()

            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            self.lpips_loss.load_state_dict(torch.load(os.path.join(self.model_dir, "lpips_vgg.pth"), map_location=self.device))
            self.lpips_loss.eval()
            print("✅ 모든 모델 로드 완료.")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")

    # [수정] 헬퍼 메서드: 이미지를 Base64로 변환
    def encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
        except:
            return ""

    def generate_heatmap(self, original_tensor, recon_tensor, original_image_path, save_filename):
        diff = torch.abs(original_tensor - recon_tensor)
        diff_map = diff.mean(dim=1).squeeze().cpu().numpy()

        fake_intensity = np.clip((THRESHOLD - diff_map) / THRESHOLD, 0, 1)
        fake_intensity = np.power(fake_intensity, 2.5) 
        fake_intensity = (fake_intensity * 255).astype(np.uint8)

        original_img = cv2.imread(original_image_path)
        h, w = original_img.shape[:2]
        fake_intensity_resized = cv2.resize(fake_intensity, (w, h))

        heatmap = cv2.applyColorMap(fake_intensity_resized, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)

        save_path = os.path.join(UPLOAD_DIR, f"heatmap_{save_filename}")
        cv2.imwrite(save_path, overlay)
        
        # Base64 문자열 반환
        return self.encode_image_to_base64(save_path)

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
        except:
            return 0.5

    def get_score(self, img_path):
        try:
            complexity = self.get_complexity(img_path)
            base_threshold = THRESHOLD 
            adaptive_threshold = base_threshold * (image_min_ratio + (1 - image_min_ratio) * complexity)

            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            min_error = float('inf')
            best_recon = None

            with torch.no_grad():
                for name, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    loss = self.lpips_loss(img_tensor, recon).item()
                    if loss < min_error:
                        min_error = loss
                        best_recon = recon
            
            # 히트맵 데이터 생성
            heatmap_data = self.generate_heatmap(img_tensor, best_recon, img_path, os.path.basename(img_path))
            
            # [추가] 원본 이미지 데이터 생성
            original_data = self.encode_image_to_base64(img_path)

            return min_error, complexity, adaptive_threshold, heatmap_data, original_data

        except Exception as e:
            print(f"❌ 분석 중 에러 발생 ({img_path}): {e}")
            return 999.0, 0.5, THRESHOLD, None, None
        
    def get_complexity_from_frame(self, frame_cv2):
        try:
            gray = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except:
            return 0.5

    def process_video(self, video_path, target_samples=30):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return 999.0, 0.5, THRESHOLD, "파일 오류", None, None

        dynamic_sample_rate = max(1, total_frames // target_samples)
        scores, complexities = [], []
        current_frame, extracted_count = 0, 0
        best_recon_frame, min_video_error = None, float('inf')
        sample_tensor = None
        best_frame_raw = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if current_frame % dynamic_sample_rate == 0:
                comp = self.get_complexity_from_frame(frame)
                complexities.append(comp)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    current_min = float('inf')
                    current_recon = None
                    for vae in self.vaes.values():
                        recon = vae(img_tensor).sample
                        err = self.lpips_loss(img_tensor, recon).item()
                        if err < current_min:
                            current_min = err
                            current_recon = recon
                    
                    scores.append(current_min)
                    if current_min < min_video_error:
                        min_video_error = current_min
                        best_recon_frame = current_recon
                        sample_tensor = img_tensor
                        best_frame_raw = frame.copy()

                extracted_count += 1
                if extracted_count >= target_samples: break
            current_frame += 1
        cap.release()
        
        if not scores: return 999.0, 0.5, THRESHOLD, "분석 불가", None, None
        
        avg_score = sum(scores) / len(scores)
        avg_complexity = sum(complexities) / len(complexities)
        adaptive_threshold = THRESHOLD * (video_min_ratio + (1 - video_min_ratio) * avg_complexity)
        verdict = "가짜" if avg_score < adaptive_threshold else "진짜"

        heatmap_data = None
        original_data = None
        if best_recon_frame is not None:
            ts = int(time.time())
            temp_frame_path = os.path.join(UPLOAD_DIR, f"temp_{ts}.jpg")
            cv2.imwrite(temp_frame_path, best_frame_raw)
            heatmap_data = self.generate_heatmap(sample_tensor, best_recon_frame, temp_frame_path, f"{ts}_video.jpg")
            # 대표 프레임을 원본 이미지로 사용
            original_data = self.encode_image_to_base64(temp_frame_path)

        return avg_score, avg_complexity, adaptive_threshold, verdict, heatmap_data, original_data

# ================= 3. FastAPI 서버 설정 =================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

scanner = AerobladeScanner(MODEL_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "index.html 파일을 찾을 수 없습니다."

@app.post("/upload")
async def detect_files(
    files: List[UploadFile] = File(...),
    expected_answer: str = Form("none") 
):
    results = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    total_files = len(files)
    correct_count = 0

    for file in files:
        ts = int(time.time() * 1000)
        unique_filename = f"{ts}_{file.filename}"
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        current_threshold, complexity, method = THRESHOLD, 0.5, "알 수 없음"
        heatmap_data = None
        original_data = None # [추가] 원본 데이터 변수

        if file_ext in video_extensions:
            # 반환값 6개로 수정
            score, complexity, current_threshold, verdict, heatmap_data, original_data = scanner.process_video(file_path)
            method = "AEROBLADE (영상 분석)"
        else:
            has_watermark = scanner.check_digital_traces(file_path)
            if has_watermark:
                verdict, score, complexity, current_threshold, method = "가짜", 0.0, 0.5, 0.0, "메타데이터 흔적"
                original_data = scanner.encode_image_to_base64(file_path)
            else:
                # 반환값 5개로 수정
                score, complexity, current_threshold, heatmap_data, original_data = scanner.get_score(file_path)
                verdict = "가짜" if score < current_threshold else "진짜"
                method = "AEROBLADE (이미지 분석)"
        
        is_correct = None
        if expected_answer in ["진짜", "가짜"]:
            is_correct = (verdict == expected_answer)
            if is_correct: correct_count += 1

        print(f"[{file.filename}] \n 판별 결과: {verdict} 점수: {score:.5f}, 방법: {method}, \n 적용 임계값: {current_threshold:.5f}), 복잡도: {complexity:.3f}")

        results.append({
            "filename": file.filename, 
            "result": verdict, 
            "score": round(score, 5),
            "threshold_used": round(current_threshold, 5), 
            "complexity": round(complexity, 3), 
            "method": method,
            "is_correct": is_correct,
            "heatmap_url": heatmap_data,
            "original_url": original_data # [필드 추가] 원본 이미지를 Base64로 전송
        })
        
    if results:
        df = pd.DataFrame(results)
        csv_path = "detection_history.csv"
        
        # 파일이 없으면 새로 만들고(header 포함), 있으면 내용만 추가(mode='a')
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
        
    accuracy_percent = (correct_count / total_files * 100) if expected_answer != "none" and total_files > 0 else 0.0

    return {
        "summary": {
            "total": total_files,
            "correct": correct_count,
            "accuracy": round(accuracy_percent, 2),
            "tested_type": expected_answer
        },
        "details": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408)