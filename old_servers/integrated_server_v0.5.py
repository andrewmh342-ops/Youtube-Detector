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
from torchvision.models import efficientnet_b0

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
class EfficientNetScanner:
    def __init__(self, model_path, device=DEVICE):
        self.device = device
        self.model = efficientnet_b0()
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_score(self, img_path):
        """이미지의 가짜 확률(0~1)을 반환"""
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            # pred 1이 Deepfake라고 가정 (web-app.py 기준)
            fake_prob = probs[1].item() 
        return fake_prob
    
class LOTAScanner:
    def extract_lsb_noise(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        
        # 하위 3개 비트 플레인 추출 (x0, x1, x2)
        # 수식: z = 2^2 * x2 + 2 * x1 + x0
        lsb_img = img & 7 # 0~7 사이의 노이즈 맵 추출
        
        # 노이즈 신호 증폭 (LOTA 정규화)
        lsb_img = cv2.normalize(lsb_img, None, 0, 255, cv2.NORM_MINMAX)
        return lsb_img

    def get_noise_score(self, img_path):
        lsb_map = self.extract_lsb_noise(img_path)
        if lsb_map is None: return 0.5
        
        # 실제 사진은 조밀하고 일관된 노이즈 군집을 형성함
        # 생성 이미지는 상위 비트의 시맨틱 '누출(Leakage)'로 인해 불규칙함
        # 여기서는 단순 분산/표준편차로 노이즈의 불규칙성을 측정 (LOTA NBC 간략화)
        std_dev = np.std(lsb_map)
        return float(std_dev / 128.0) # 노이즈 점수 정규화

def get_texture_crop(img_path, crop_size=224):
    """
    ITW-SM(2025) 연구 기반: 생성 아티팩트가 밀집된 고주파 영역(Texture)을 타겟팅하여 자름
    단순 중앙 크롭보다 탐지 AUC를 3% 이상 향상시킴
    """
    img = cv2.imread(img_path)
    if img is None: return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 다방향 그래디언트(Laplacian) 연산으로 텍스처 밀집도 계산
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    h, w = gray.shape
    if h < crop_size or w < crop_size:
        return cv2.resize(img, (crop_size, crop_size))

    # 이미지를 그리드로 나누어 가장 '복잡한(Texture)' 패치 선택
    best_val = -1
    best_patch = None
    
    # 3x3 그리드에서 텍스처 탐색 (MGPS 전략 시뮬레이션)
    for y in range(0, h - crop_size, crop_size // 2):
        for x in range(0, w - crop_size, crop_size // 2):
            patch_var = np.var(laplacian[y:y+crop_size, x:x+crop_size])
            if patch_var > best_val:
                best_val = patch_var
                best_patch = img[y:y+crop_size, x:x+crop_size]
    
    return best_patch if best_patch is not None else img[:crop_size, :crop_size]

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
            'sd1.5': 0.040,
            'sd2.1': 0.035,
            'sdxl': 0.025,
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
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is None: return 0.5
            
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            complexity = np.clip(np.log1p(laplacian_var) / 7.0, 0, 1)
            return float(complexity)
        except Exception as e:
            print(f"복잡도 계산 실패: {e}")
            return 0.5

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

    def process_video(self, video_path, target_samples=30):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return 999.0, 0.5, 0.055, "파일 오류", "None", 0.0 # 반환 개수 맞춤

        dynamic_sample_rate = max(1, total_frames // target_samples)
        
        model_total_errors = {name: 0.0 for name in self.vaes.keys()}
        complexities = []
        eff_scores_sum = 0.0  # <--- [수정] 변수 초기화 추가
        extracted_count = 0
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame % dynamic_sample_rate == 0:
                complexities.append(self.get_complexity_from_frame(frame))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # EfficientNet 점수 누적
                temp_path = f"temp_frame_{extracted_count}.jpg"
                cv2.imwrite(temp_path, frame)
                eff_scores_sum += self.eff_scanner.get_score(temp_path) # <--- 전역 객체 eff_scanner 사용
                if os.path.exists(temp_path): os.remove(temp_path)
                
                # Aeroblade 점수 누적
                with torch.no_grad():
                    for name, vae in self.vaes.items():
                        recon = vae(img_tensor).sample
                        loss = self.lpips_loss(img_tensor, recon).item()
                        model_total_errors[name] += loss
                
                extracted_count += 1
                if extracted_count >= target_samples: break
            current_frame += 1
        cap.release()
        
        # 모델별 평균 및 최종 데이터 계산
        avg_errors = {name: total / extracted_count for name, total in model_total_errors.items()}
        detected_model = min(avg_errors, key=avg_errors.get)
        avg_score = avg_errors[detected_model]
        avg_complexity = sum(complexities) / len(complexities)
        avg_eff_prob = eff_scores_sum / extracted_count # <--- [수정] EfficientNet 평균 확률 계산

        base_thr = self.base_thresholds.get(detected_model, 0.055)
        adaptive_threshold = base_thr * (video_min_ratio + (1 - video_min_ratio) * avg_complexity)
        
        verdict = "가짜" if avg_score < adaptive_threshold else "진짜"
        # avg_eff_prob를 함께 반환하도록 수정
        return avg_score, avg_complexity, adaptive_threshold, verdict, detected_model, avg_eff_prob

# ================= 3. FastAPI 서버 설정 =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = AerobladeScanner(MODEL_DIR)
eff_scanner = EfficientNetScanner("local_models/best_model-v3.pt")

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

        file_ext = os.path.splitext(file.filename)[1].lower()
        method = ""
        aero_res = {"score": 0.0, "threshold": 0.0, "is_fake": False}
        eff_res = {"prob": 0.0, "is_fake": False}
        complexity = 0.5
        detected_model = "N/A"

        if file_ext in video_extensions:
            method = "Ensemble (Video)"
            # [수정] process_video에서 eff_prob까지 받아옴
            aero_score, complexity, aero_thr, aero_verdict, detected_model, eff_prob = scanner.process_video(file_path)
            aero_res = {"score": aero_score, "threshold": aero_thr, "is_fake": (aero_verdict == "가짜")}
            eff_res = {"prob": eff_prob, "is_fake": eff_prob > 0.45}
        else:
            has_watermark = scanner.check_digital_traces(file_path)
            if has_watermark:
                method = "Metadata Trace"
                aero_res["is_fake"] = True # 메타데이터 걸리면 강제 가짜 처리
                detected_model = "Metadata"
            else:
                method = "Ensemble (Image)"
                aero_score, complexity, aero_thr, detected_model = scanner.get_score(file_path)
                aero_res = {"score": aero_score, "threshold": aero_thr, "is_fake": aero_score < aero_thr}
                eff_prob = eff_scanner.get_score(file_path)
                eff_res = {"prob": eff_prob, "is_fake": eff_prob > 0.45}

        # --- 앙상블 판정 로직 (호준님 전략 반영) ---
        is_fake_aero = aero_res["is_fake"]
        eff_fake_prob = eff_res["prob"]

        aero_conf = max(0, (aero_thr - aero_score) / aero_thr) if aero_thr > 0 else 0
        combined_score = (aero_conf * 0.4) + (eff_fake_prob * 0.6)

        final_verdict = "진짜" 
        if combined_score > 0.55:
            final_verdict = "가짜"
        elif combined_score > 0.45:
            # 경계 구간: EfficientNet이 더 확신할 때만 가짜로 판정
            final_verdict = "가짜" if eff_fake_prob > 0.60 else "진짜"
        else:
            final_verdict = "진짜"

        ensemble_confidence = (eff_fake_prob * 0.7) + (0.3 if is_fake_aero else 0)

        # --- 3. 결과 정리 및 로깅 ---
        print(f"[{file.filename}] 판별: {final_verdict}, Ensemble Confidence: {ensemble_confidence:.4f},탐지된 모델: {detected_model}")
        print(f"Aeroblade: {aero_res['is_fake']}, aero_score: {aero_res['score']}, aero_thr: {aero_res['threshold']},aero_complexity: {complexity}")
        print(f"EffNet: {eff_res['is_fake']}, eff_prob: {eff_res['prob']})")
        
        res_obj = {
                "filename": file.filename,
                "result": final_verdict,
                "ensemble_confidence": round(ensemble_confidence, 4),
                "detected_model": detected_model,
                "aero_is_fake": aero_res['is_fake'],
                "aero_score": round(aero_res['score'], 5),
                "aero_thr": round(aero_res['threshold'], 5),
                "aero_complexity": round(complexity, 4),
                "eff_is_fake": eff_res['is_fake'],
                "eff_prob": round(eff_res['prob'], 4),
                "method": method
            }
        results.append(res_obj)

    if results:
        df = pd.DataFrame(results)
        csv_path = "detection_history.csv"
        df.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path), encoding='utf-8-sig')
            
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20408) # 로컬 호스트에서 실행 host="127.0.0.1" | 외부 접속 허용 host="0.0.0.0"