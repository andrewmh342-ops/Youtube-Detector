import os
import math
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
import sqlite3
from datetime import datetime
import uvicorn
import shutil
import base64
import time
from typing import List

# ================= 1. 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
IMAGE_MIN_RATIO = 0.8
BLACKLIST_THRESHOLD = 3

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ================= 2. 통합 탐지 스캐너 클래스 =================
class IntegratedScanner:
    def __init__(self, model_dir, device=DEVICE):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
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
            'sd1.5': 0.065,
            'sd2.1': 0.060,
            'sdxl': 0.045,
            'sd3': 0.012
        }
        
        self.r_low = 0.05
        self.r_high = 0.50

    def _prepare_models(self):
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        models = {
            "vae_sd1.5": ("stabilityai/sd-vae-ft-mse", None),
            "vae_sd2.1": ("stabilityai/sd-vae-ft-ema", None),
            "vae_sdxl": ("stabilityai/sdxl-vae", None)
        }
        for folder, (hub_path, subfolder) in models.items():
            path = os.path.join(self.model_dir, folder)
            if not os.path.exists(path):
                if subfolder:
                    AutoencoderKL.from_pretrained(hub_path, subfolder=subfolder).save_pretrained(path)
                else:
                    AutoencoderKL.from_pretrained(hub_path).save_pretrained(path)

        lpips_path = os.path.join(self.model_dir, "lpips_vgg.pth")
        if not os.path.exists(lpips_path):
            lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
            torch.save(lpips_model.state_dict(), lpips_path)

    def _load_models(self):
        try:
            load_args = {"torch_dtype": self.dtype, "use_safetensors": True}
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd1.5"), **load_args).to(self.device).eval()
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sd2.1"), **load_args).to(self.device).eval()
            self.vaes['sdxl'] = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sdxl"), **load_args).to(self.device).eval()

            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device).eval()
            self.lpips_loss.load_state_dict(torch.load(os.path.join(self.model_dir, "lpips_vgg.pth"), map_location=self.device))
            print("✅ 모든 모델 로드 완료.")
        except Exception as e: print(f"❌ 로딩 실패: {e}")

    def encode_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
        except: return ""

    def generate_heatmap(self, original_tensor, recon_tensor, threshold, save_filename):
        """재구성 오차를 기반으로 히트맵 생성"""
        diff = torch.abs(original_tensor - recon_tensor)
        diff_map = diff.mean(dim=1).squeeze().cpu().to(torch.float32).numpy()

        # 오차가 적을수록(AI 재구성이 잘 될수록) 더 붉게 표시
        fake_intensity = np.clip((threshold - diff_map) / (threshold + 1e-8), 0, 1)
        fake_intensity = np.power(fake_intensity, 2.5) 
        fake_intensity = (fake_intensity * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(fake_intensity, cv2.COLORMAP_HOT)
        save_path = os.path.join(UPLOAD_DIR, f"heatmap_{save_filename}.jpg")
        cv2.imwrite(save_path, heatmap)
        
        return self.encode_image_to_base64(save_path)

    def get_texture_crop(self, pil_img, crop_size=512):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < crop_size or w < crop_size:
            return pil_img.resize((crop_size, crop_size), Image.LANCZOS)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        best_score, best_coords, stride = -1, (0, 0), 64 
        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                score = np.sum(laplacian[y:y+crop_size, x:x+crop_size])
                if score > best_score:
                    best_score, best_coords = score, (y, x)
        y, x = best_coords
        return pil_img.crop((x, y, x + crop_size, y + crop_size))

    def get_fire_score(self, img_tensor, recon_tensor):
        img_fft = torch.fft.fftshift(torch.fft.fftn(img_tensor.float(), dim=(-2, -1)))
        recon_fft = torch.fft.fftshift(torch.fft.fftn(recon_tensor.float(), dim=(-2, -1)))
        b, c, h, w = img_tensor.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = torch.sqrt((y - h//2)**2 + (x - w//2)**2).to(self.device)
        mask = (dist > (self.r_low * h)) & (dist < (self.r_high * h))
        diff_fft = torch.abs(img_fft - recon_fft)
        return ((diff_fft * mask).sum() / (mask.sum() + 1e-8)).item()

    def get_complexity(self, image_pil):
        cv_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(cv_img, cv2.CV_64F).var()
        return float(np.clip(np.log1p(laplacian_var) / 7.0, 0, 1))

    #메타데이터 분석 함수
    def analyze_metadata(self, img_path):
        results = {
            "is_ai_metadata": False,
            "source": "Unknown",
            "details": ""
        }
        
        try:
            with Image.open(img_path) as img:
                # 1. 챗GPT (DALL-E 3) & C2PA 확인
                # C2PA 데이터는 보통 'JUMBF' 유닛에 저장됩니다.
                if "XML:com.adobe.xmp" in img.info:
                    xmp_data = img.info["XML:com.adobe.xmp"]
                    if "dalle" in xmp_data.lower() or "openai" in xmp_data.lower():
                        results.update({"is_ai_metadata": True, "source": "DALL-E 3 (OpenAI)", "details": "C2PA/XMP 데이터 감지"})

                # 2. 제미나이 (Google / IPTC 확인)
                # 구글은 IPTC의 DigitalSourceType 필드에 AI 정보를 기록합니다.
                iptc = img.info.get("iptc")
                if iptc:
                    # 'trainedAlgorithmicMedia' 등의 키워드 확인
                    if b"google" in str(iptc).lower() or b"imagen" in str(iptc).lower():
                        results.update({"is_ai_metadata": True, "source": "Gemini (Google)", "details": "IPTC 디지털 소스 정보 확인"})

                # 3. 그록 (Grok / Flux.1 PNG Info 확인)
                # Flux 모델은 PNG의 'tEXt' 청크에 프롬프트와 모델 정보를 남기는 경우가 많습니다.
                if img.format == "PNG":
                    for key, value in img.info.items():
                        if any(word in str(value).lower() for word in ["flux", "grok", "xai"]):
                            results.update({"is_ai_metadata": True, "source": "Grok (xAI)", "details": f"PNG Info 내 {key} 태그 감지"})

        except Exception as e:
            print(f"메타데이터 분석 오류: {e}")
        
        return results

    def get_score(self, img_path_or_pil, filename="temp"):
        try:
            metadata_res = self.analyze_metadata(img_path_or_pil)
            image = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil
            complexity = self.get_complexity(image)
            image_crop = self.get_texture_crop(image, crop_size=512)
            img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device, dtype=self.dtype)

            best_recon, min_ae, detected_model = None, float('inf'), "sd1.5"
            fire_score = 0.0

            with torch.inference_mode():
                for name, vae in self.vaes.items():
                    latents = vae.encode(img_tensor).latent_dist.sample()
                    recon = vae.decode(latents).sample
                    ae_val = self.lpips_loss(img_tensor.to(torch.float32), recon.to(torch.float32)).item()
                    if ae_val < min_ae:
                        min_ae, best_recon, detected_model = ae_val, recon, name
                fire_score = self.get_fire_score(img_tensor, best_recon)
            
            base_thr = self.base_thresholds.get(detected_model, 0.050)
            adaptive_thr = base_thr * (IMAGE_MIN_RATIO + (1 - IMAGE_MIN_RATIO) * complexity)
            
            # 히트맵 및 원본 전송용 데이터 생성
            ts = int(time.time() * 1000)
            temp_orig_path = os.path.join(UPLOAD_DIR, f"orig_{ts}_{filename}.jpg")
            image_crop.save(temp_orig_path)
            
            original_base64 = self.encode_image_to_base64(temp_orig_path)
            heatmap_base64 = self.generate_heatmap(img_tensor, best_recon, adaptive_thr, f"{ts}_{filename}")
            if metadata_res["is_ai_metadata"]:
                verdict = f"가짜 ({metadata_res['source']})"
            elif fire_score < 35.0:
                verdict = "가짜"
            else:
                verdict = "가짜" if (min_ae < adaptive_thr) or (fire_score < 42.0) else "진짜"

            return min_ae, fire_score, complexity, adaptive_thr, detected_model, original_base64, heatmap_base64, verdict
        except Exception as e:
            print(f"오류: {e}")
            return 999.0, 0.0, 0.5, 0.05, "Error", "", ""

    def process_video(self, video_path, target_samples=15):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened(): return 999.0, 0.0, 0.5, 0.05, "Error", "", ""

        ae_list, fire_list, comp_list, thr_list = [], [], [], []
        best_frame_data = {"score": float('inf'), "orig": "", "heat": ""}
        
        sample_interval = max(1, total_frames // target_samples)
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # [중요] 변수 8개를 모두 받아야 에러가 나지 않습니다.
            ae, fire, comp, thr, model, orig, heat, _ = self.get_score(pil_img, "video_frame")
            
            ae_list.append(ae); fire_list.append(fire); comp_list.append(comp); thr_list.append(thr)
            if ae < best_frame_data["score"]:
                best_frame_data.update({"score": ae, "orig": orig, "heat": heat})
            if len(ae_list) >= target_samples: break
        cap.release()

        avg_ae = sum(ae_list)/len(ae_list)
        avg_fire = sum(fire_list)/len(fire_list)
        # 비디오는 물리적 점수의 평균으로 최종 판정
        verdict = "가짜" if (avg_ae < sum(thr_list)/len(thr_list)) or (avg_fire < 42.0) else "진짜"
        return avg_ae, avg_fire, sum(comp_list)/len(comp_list), sum(thr_list)/len(thr_list), verdict, best_frame_data["orig"], best_frame_data["heat"]

# ================= 3. FastAPI 서버 =================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
scanner = IntegratedScanner(MODEL_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
# DB 초기화
def init_db():
    conn = sqlite3.connect("fake_sites.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fake_sites (
            url TEXT PRIMARY KEY,
            detected_at TEXT,
            hit_count INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

init_db()
scanner = IntegratedScanner("local_models")

# 관리자 대시보드 - DB에 등록된 사이트 목록 확인용
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    conn = sqlite3.connect("fake_sites.db")
    cursor = conn.cursor()
    cursor.execute("SELECT url, detected_at, hit_count FROM fake_sites ORDER BY detected_at DESC")
    rows = cursor.fetchall()
    conn.close()

    html_content = f"""
    <html>
        <head><title>Deepfake Detector Admin</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
            <h2>🚫 탐지된 가짜 이미지 사이트 목록</h2>
            <table border="1" style="width:100%; border-collapse: collapse;">
                <tr style="background: #eee;">
                    <th>도메인 URL</th><th>최근 탐지 시간</th><th>탐지 횟수</th>
                </tr>
                {"".join([f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>" for r in rows])}
            </table>
        </body>
    </html>
    """
    return html_content
    
# 가짜 사이트 등록 API
@app.post("/report-fake-url")
async def report_fake_url(url: str = Form(...)):
    clean_url = url.strip().rstrip('/')
    conn = sqlite3.connect("fake_sites.db")
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO fake_sites (url, detected_at, hit_count) 
            VALUES (?, ?, 1)
            ON CONFLICT(url) DO UPDATE SET hit_count = hit_count + 1
        ''', (clean_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return {"status": "success", "url": clean_url}
    except Exception as e: 
        return {"status": "error", "message": str(e)}
    finally: 
        conn.close()

# 사이트 조회 API
@app.get("/check-site")
async def check_site(url: str):
    conn = sqlite3.connect("fake_sites.db")
    cursor = conn.cursor()
    cursor.execute("SELECT hit_count FROM fake_sites WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()

    hit_count = result[0] if result else 0
    is_blacklisted = hit_count >= BLACKLIST_THRESHOLD

    return {"is_blacklisted": is_blacklisted, "hit_count": hit_count, "threshold": BLACKLIST_THRESHOLD}

# 파일 업로드 API
@app.post("/upload")
async def detect_files(
    files: List[UploadFile] = File(...), 
    expected_answer: str = Form("none")
):
    results = []
    correct_count = 0
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    for file in files:
        # 파일 저장
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as buffer: 
            shutil.copyfileobj(file.file, buffer)
        
        ext = os.path.splitext(file.filename)[1].lower()
        
        # 분석 실행
        if ext in video_exts:
            ae, fire, comp, thr, verdict, orig, heat = scanner.process_video(path)
            method = "AEROBLADE+FIRE (Video)"
            detected_model = "Ensemble"
        else:
            ae, fire, comp, thr, model, orig, heat, verdict = scanner.get_score(path, file.filename)
            method = f"AEROBLADE ({model})"
            detected_model = model

        # 정답 테스트 모드 계산
        is_correct = None
        if expected_answer in ["진짜", "가짜"]:
            is_correct = (verdict == expected_answer)
            if is_correct: correct_count += 1

        # 결과 리스트에 모든 지표 추가 (HTML 출력 및 CSV 저장용)
        res_entry = {
            "filename": file.filename,
            "result": verdict,
            "score": round(ae, 5),          # LPIPS Error (HTML 표시용)
            "ae_score": round(ae, 5),       # CSV 상세 저장용
            "fire_score": round(fire, 5),
            "threshold": round(thr, 5),     # 적용된 임계값
            "complexity": round(comp, 3),   # 시각적 복잡도
            "method": method,
            "detected_source": detected_model,
            "is_correct": is_correct,
            "original_url": orig,
            "heatmap_url": heat
        }
        results.append(res_entry)

        print(f"[{file.filename}] 결과: {verdict} (AE: {ae:.5f}, FIRE: {fire:.2f}, Thr: {thr:.5f})")

        # CSV 로깅 로직
        if results:
            log_df = pd.DataFrame([{k: v for k, v in r.items() if 'url' not in k} for r in results])
            csv_path = "detection_history.csv"
            if not os.path.exists(csv_path):
                log_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            else:
                log_df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

    # 요약 정보와 함께 반환
    accuracy = round((correct_count / len(files)) * 100, 2) if expected_answer != "none" else 0
    return {
        "summary": {
            "total": len(files),
            "correct": correct_count,
            "accuracy": accuracy,
            "tested_type": expected_answer
        },
        "details": results
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=80,
    )