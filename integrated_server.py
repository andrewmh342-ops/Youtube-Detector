import os
import io
import uuid
import json
import magic
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from diffusers import AutoencoderKL
import lpips
import piexif
import c2pa
from pillow_heif import register_heif_opener
register_heif_opener()
from fastapi.responses import HTMLResponse
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from datetime import datetime
import uvicorn
import base64
import time
from pathlib import Path
from typing import List

# ================= 설정 및 경로 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "local_models"
UPLOAD_DIR = "uploads"
IMAGE_MIN_RATIO = 0.8
BLACKLIST_THRESHOLD = 3

# 서버 호스트 및 포트 설정 (필요 시 변경: 로컬 테스트=localhost, 외부 접근=0.0.0.0)
set_host = "0.0.0.0"
set_port = 8080

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ================= 파일 업로드 제한 =================
MAX_FILE_SIZE   = 50 * 1024 * 1024   # 파일 1개당 최대 50MB
MAX_FILES       = 50                  # 요청당 최대 파일 개수

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}

IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
}

VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}

# ================= 파일 검증 유틸 함수 =================

def make_safe_filename(original_filename: str) -> str:
    ext = Path(original_filename).suffix.lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif",
                    ".mp4", ".mov", ".avi", ".mkv", ".webm"}
    safe_ext = ext if ext in allowed_exts else ""
    return f"{uuid.uuid4().hex}{safe_ext}"


async def validate_and_read_file(file: UploadFile) -> tuple[bytes, str]:
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"[{file.filename}] 파일 크기가 {MAX_FILE_SIZE // (1024 * 1024)}MB를 초과합니다."
        )
    try:
        real_mime = magic.from_buffer(content[:2048], mime=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"[{file.filename}] 파일 형식을 판별할 수 없습니다."
        )

    if real_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"[{file.filename}] 허용되지 않는 파일 형식입니다. (감지된 형식: {real_mime})"
        )

    if real_mime in IMAGE_MIME_TYPES:
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
            img = Image.open(io.BytesIO(content))
            img.load()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"[{file.filename}] 유효하지 않거나 손상된 이미지 파일입니다."
            )

    return content, real_mime

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
            self.vaes['sdxl']  = AutoencoderKL.from_pretrained(os.path.join(self.model_dir, "vae_sdxl"),  **load_args).to(self.device).eval()

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
        diff = torch.abs(original_tensor - recon_tensor)
        diff_map = diff.mean(dim=1).squeeze().cpu().to(torch.float32).numpy()

        # 오차가 적을수록 더 붉게 표시
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
        img_fft   = torch.fft.fftshift(torch.fft.fftn(img_tensor.float(),   dim=(-2, -1)))
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

    # C2PA 서명 탐지
    def _detect_c2pa(self, img_path: str) -> dict:
        result = {"detected": False, "source": "", "details": ""}
        if not isinstance(img_path, str):
            return result
        try:
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".webp": "image/webp"}
            ext = os.path.splitext(img_path)[1].lower()
            mime = mime_map.get(ext)
            if not mime:
                return result

            reader = c2pa.Reader(mime, open(img_path, "rb"))
            manifest_json = reader.json()
            manifest = json.loads(manifest_json)

            active_label = manifest.get("active_manifest", "")
            manifests = manifest.get("manifests", {})
            if not active_label or active_label not in manifests:
                return result

            active = manifests[active_label]
            assertions = active.get("assertions", [])

            for assertion in assertions:
                label = assertion.get("label", "").lower()
                data  = assertion.get("data", {})

                if "generative" in label or "ai.generated" in label:
                    entries = data if isinstance(data, list) else [data]
                    for entry in entries:
                        gen_tool = str(entry.get("alg", "") + entry.get("name", "")).lower()
                        source = self._map_c2pa_source(gen_tool, active)
                        result.update({"detected": True, "source": source,
                                       "details": f"C2PA AI 생성 서명 확인 (assertion: {label})"})
                        return result

                if "digitalsource" in label or "digital_source" in label:
                    src_type = str(data.get("digitalSourceType", "")).lower()
                    if "trainedalgorithmicmedia" in src_type or "compositewithtrainedalgorithmicmedia" in src_type:
                        source = self._map_c2pa_source("", active)
                        result.update({"detected": True, "source": source,
                                       "details": f"C2PA digitalSourceType: {src_type}"})
                        return result

                if "actions" in label:
                    for action in data.get("actions", []):
                        src_type = str(action.get("digitalSourceType", "")).lower()
                        if "trainedalgorithmicmedia" in src_type or "compositewithtrainedalgorithmicmedia" in src_type:
                            agent = action.get("softwareAgent", {})
                            agent_name = agent.get("name", "") if isinstance(agent, dict) else str(agent)
                            source = self._map_c2pa_source(agent_name.lower(), active)
                            result.update({"detected": True, "source": source,
                                           "details": f"C2PA actions.digitalSourceType: trainedAlgorithmicMedia (agent: {agent_name})"})
                            return result

                    C2PA_VENDOR_KEYWORDS = {
                        "google": "Imagen (Google)", "gemini": "Gemini (Google)",
                        "openai": "DALL-E (OpenAI)", "chatgpt": "DALL-E (OpenAI)",
                        "midjourney": "Midjourney", "stability": "Stable Diffusion",
                        "adobe": "Firefly (Adobe)", "firefly": "Firefly (Adobe)",
                    }
                    actions_text = str(data).lower()
                    for kw, source in C2PA_VENDOR_KEYWORDS.items():
                        if kw in actions_text:
                            result.update({"detected": True, "source": source,
                                           "details": f"C2PA actions 내 '{kw}' 감지"})
                            return result

            claim_generator = active.get("claim_generator", "").lower()
            if any(kw in claim_generator for kw in ["openai", "adobe", "google", "midjourney", "stability"]):
                source = self._map_c2pa_source(claim_generator, active)
                result.update({"detected": True, "source": source,
                               "details": f"C2PA claim_generator: {claim_generator}"})

        except Exception as e:
            pass
        return result

    def _map_c2pa_source(self, gen_tool: str, manifest_data: dict) -> str:
        claim = (manifest_data.get("claim_generator", "") + gen_tool).lower()
        if "openai" in claim or "dalle" in claim or "chatgpt" in claim: return "DALL-E (OpenAI)"
        if "google" in claim or "imagen" in claim:  return "Imagen (Google)"
        if "adobe" in claim or "firefly" in claim:  return "Firefly (Adobe)"
        if "midjourney" in claim:                   return "Midjourney"
        if "stability" in claim or "stable" in claim: return "Stable Diffusion"
        return "AI 생성 (C2PA 서명 확인됨)"

    # 메타데이터 심층 분석
    def _detect_metadata(self, img_path_or_pil) -> dict:
        result = {"detected": False, "source": "", "details": ""}

        AI_KEYWORDS = {
            "openai": "DALL-E (OpenAI)", "dall-e": "DALL-E (OpenAI)", "dalle": "DALL-E (OpenAI)",
            "imagen": "Imagen (Google)", "google imagen": "Imagen (Google)",
            "gemini": "Gemini (Google)",
            "midjourney": "Midjourney",
            "niji journey": "Midjourney (Niji)",
            "stable diffusion": "Stable Diffusion", "stability ai": "Stable Diffusion",
            "black forest labs": "FLUX (Black Forest Labs)",
            "grok": "Grok (xAI)",
            "adobe firefly": "Firefly (Adobe)", "firefly": "Firefly (Adobe)",
            "ideogram": "Ideogram", "leonardo.ai": "Leonardo.Ai",
            "runwayml": "RunwayML", "runway ml": "RunwayML",
            "pika labs": "Pika Labs",
            "kling": "Kling (Kuaishou)", "hailuo": "Hailuo (MiniMax)",
        }

        try:
            if isinstance(img_path_or_pil, str):
                pil_img = Image.open(img_path_or_pil)
                img_path = img_path_or_pil
            else:
                pil_img = img_path_or_pil
                img_path = None

            info = pil_img.info or {}

            all_text = " ".join(str(v) for v in info.values()).lower()
            for kw, source in AI_KEYWORDS.items():
                if kw in all_text:
                    result.update({"detected": True, "source": source,
                                   "details": f"PNG 메타데이터 내 '{kw}' 감지"})
                    return result

            xmp_data = info.get("XML:com.adobe.xmp", "")
            if xmp_data:
                xmp_lower = xmp_data.lower()
                for kw, source in AI_KEYWORDS.items():
                    if kw in xmp_lower:
                        result.update({"detected": True, "source": source,
                                       "details": f"XMP 블록 내 '{kw}' 감지"})
                        return result
                if "trainedAlgorithmicMedia" in xmp_data or "compositeWithTrainedAlgorithmicMedia" in xmp_data:
                    result.update({"detected": True, "source": "AI 생성 (XMP IPTC4XMP)",
                                   "details": "XMP DigitalSourceType=trainedAlgorithmicMedia 확인"})
                    return result

            if img_path and pil_img.format == "JPEG":
                try:
                    exif_dict = piexif.load(img_path)
                    for ifd_name, ifd in exif_dict.items():
                        if ifd_name == "GPS" or not isinstance(ifd, dict):
                            continue
                        for tag_val in ifd.values():
                            tag_str = tag_val.decode("utf-8", errors="ignore").lower() \
                                if isinstance(tag_val, bytes) else str(tag_val).lower()
                            for kw, source in AI_KEYWORDS.items():
                                if kw in tag_str:
                                    result.update({"detected": True, "source": source,
                                                   "details": f"EXIF 태그 내 '{kw}' 감지"})
                                    return result
                except Exception:
                    pass

        except Exception as e:
            print(f"메타데이터 분석 오류: {e}")

        return result

    # 메타데이터 탐지 로직
    def analyze_metadata(self, img_path) -> dict:
        # C2PA 디지털 서명 탐지
        r = self._detect_c2pa(img_path)
        if r["detected"]:
            return {"is_ai_metadata": True, "source": r["source"], "details": r["details"]}

        # 메타데이터 심층 분석
        r = self._detect_metadata(img_path)
        if r["detected"]:
            return {"is_ai_metadata": True, "source": r["source"], "details": r["details"]}

        return {"is_ai_metadata": False, "source": "Unknown", "details": ""}

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
            heatmap_base64  = self.generate_heatmap(img_tensor, best_recon, adaptive_thr, f"{ts}_{filename}")

            if metadata_res["is_ai_metadata"]:
                verdict = f"가짜 ({metadata_res['source']})"
                detected_model = f"메타데이터 탐지 ({metadata_res['source']})"
            elif fire_score < 35.0:
                verdict = "가짜"
            else:
                verdict = "가짜" if (min_ae < adaptive_thr) or (fire_score < 42.0) else "진짜"

            return min_ae, fire_score, complexity, adaptive_thr, detected_model, original_base64, heatmap_base64, verdict
        except Exception as e:
            print(f"오류: {e}")
            return 999.0, 0.0, 0.5, 0.05, "Error", "", "", "진짜"

    def process_video(self, video_path, target_samples=15):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened(): return 999.0, 0.0, 0.5, 0.05, "Error", "", "", "진짜"

        ae_list, fire_list, comp_list, thr_list = [], [], [], []
        best_frame_data = {"score": float('inf'), "orig": "", "heat": ""}
        
        sample_interval = max(1, total_frames // target_samples)
        for i in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ae, fire, comp, thr, model, orig, heat, _ = self.get_score(pil_img, "video_frame")
            
            ae_list.append(ae); fire_list.append(fire); comp_list.append(comp); thr_list.append(thr)
            if ae < best_frame_data["score"]:
                best_frame_data.update({"score": ae, "orig": orig, "heat": heat})
            if len(ae_list) >= target_samples: break
        cap.release()

        avg_ae   = sum(ae_list) / len(ae_list)
        avg_fire = sum(fire_list) / len(fire_list)
        verdict  = "가짜" if (avg_ae < sum(thr_list)/len(thr_list)) or (avg_fire < 42.0) else "진짜"
        return avg_ae, avg_fire, sum(comp_list)/len(comp_list), sum(thr_list)/len(thr_list), verdict, best_frame_data["orig"], best_frame_data["heat"]


# ================= 3. FastAPI 서버 =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
scanner = IntegratedScanner(MODEL_DIR)

# 허용 경로 설정
ALLOWED_PATHS = ["/", "/upload", "/report-fake-url", "/check-site", "/blacklist"]

@app.middleware("http")
async def whitelist_filter(request: Request, call_next):
    path = request.url.path
    if path not in ALLOWED_PATHS:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return await call_next(request)

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

# 관리자 대시보드 - DB에 등록된 사이트 목록 확인용
@app.get("/blacklist", response_class=HTMLResponse)
async def list_dashboard():
    conn = sqlite3.connect("fake_sites.db")
    cursor = conn.cursor()
    cursor.execute("SELECT url, detected_at, hit_count FROM fake_sites ORDER BY detected_at DESC")
    rows = cursor.fetchall()
    conn.close()

    html_content = f"""
    <html>
        <head><title>Deepfake Detector Blacklist</title></head>
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
async def report_fake_url(url: str = Form(...), user_token: str = Form("anonymous")):
    clean_url = url.strip().rstrip('/')
    if clean_url.startswith("https://") or clean_url.startswith("http://"):
        conn = sqlite3.connect("fake_sites.db")
        cursor = conn.cursor()
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO fake_sites (url, detected_at, hit_count) 
                VALUES (?, ?, 1)
                ON CONFLICT(url) DO UPDATE SET 
                    hit_count = hit_count + 1,
                    detected_at = ?
            ''', (clean_url, now, now))
            conn.commit()
            return {"status": "success", "url": clean_url}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            conn.close()
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid URL format"})

# 사이트 조회 API
@app.get("/check-site")
async def check_site(url: str):
    conn = sqlite3.connect("fake_sites.db")
    cursor = conn.cursor()
    cursor.execute("SELECT hit_count FROM fake_sites WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()

    hit_count    = result[0] if result else 0
    is_blacklisted = hit_count >= BLACKLIST_THRESHOLD

    return {"is_blacklisted": is_blacklisted, "hit_count": hit_count, "threshold": BLACKLIST_THRESHOLD}


# 파일 업로드 API
@app.post("/upload")
async def detect_files(
    files: List[UploadFile] = File(...),
    expected_answer: str = Form("none")
):
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"한 번에 최대 {MAX_FILES}개까지만 업로드 가능합니다."
        )

    results = []
    correct_count = 0

    for file in files:
        content, real_mime = await validate_and_read_file(file)
        safe_name     = make_safe_filename(file.filename)
        display_name  = file.filename
        save_path     = os.path.join(UPLOAD_DIR, safe_name)

        with open(save_path, "wb") as f:
            f.write(content)

        is_video = real_mime in VIDEO_MIME_TYPES

        if is_video:
            ae, fire, comp, thr, verdict, orig, heat = scanner.process_video(save_path)
            method         = "AEROBLADE+FIRE (Video)"
            detected_model = "Ensemble"
        else:
            ae, fire, comp, thr, model, orig, heat, verdict = scanner.get_score(save_path, safe_name)
            method         = f"AEROBLADE ({model})"
            detected_model = model

        # 정답 테스트 모드 계산
        is_correct = None
        if expected_answer in ["진짜", "가짜"]:
            is_correct = (verdict == expected_answer)
            if is_correct:
                correct_count += 1

        res_entry = {
            "filename":        display_name,   # 원본 파일명
            "result":          verdict,
            "score":           round(ae,   5),
            "ae_score":        round(ae,   5),
            "fire_score":      round(fire, 5),
            "threshold":       round(thr,  5),
            "complexity":      round(comp, 3),
            "method":          method,
            "detected_source": detected_model,
            "is_correct":      is_correct,
            "original_url":    orig,
            "heatmap_url":     heat,
        }
        results.append(res_entry)

        print(f"[{display_name}] 결과: {verdict} (AE: {ae:.5f}, FIRE: {fire:.2f}, Thr: {thr:.5f})")

        # CSV 로깅
        if results:
            log_df   = pd.DataFrame([{k: v for k, v in r.items() if 'url' not in k} for r in results])
            csv_path = "detection_history.csv"
            if not os.path.exists(csv_path):
                log_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            else:
                log_df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

    accuracy = round((correct_count / len(files)) * 100, 2) if expected_answer != "none" else 0
    return {
        "summary": {
            "total":       len(files),
            "correct":     correct_count,
            "accuracy":    accuracy,
            "tested_type": expected_answer,
        },
        "details": results,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=set_host,
        port=set_port,
    )