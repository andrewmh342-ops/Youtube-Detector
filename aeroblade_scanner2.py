import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, PngImagePlugin
from diffusers import AutoencoderKL
import lpips
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ================= 설정 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_DIR = "local_models" # 다운로드한 모델 폴더 경로
SAVE_DIR = "local_models"        # 저장할 모델 폴더 이름

# 경로 설정
TARGET_IMG_FOLDER = "pic"        # [중요] 판별할 대상 폴더
OUTPUT_FILENAME = "result.csv"   # 결과 저장 파일명
THRESHOLD = 0.07                 # AEROBLADE 기준 점수
# ========================================

# 폴더가 없으면 생성 및 모델 다운로드 (기존 코드 유지)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"=== 모델 다운로드/확인 시작 (저장 위치: {SAVE_DIR}) ===")
try:
    # 1. SD 1.5
    path_sd15 = os.path.join(SAVE_DIR, "vae_sd1.5")
    if not (os.path.exists(path_sd15) and os.path.exists(os.path.join(path_sd15, "config.json"))):
        print("[1/3] SD 1.5 VAE 다운로드 중...")
        AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").save_pretrained(path_sd15)

    # 2. SD 2.1
    path_sd21 = os.path.join(SAVE_DIR, "vae_sd2.1")
    if not (os.path.exists(path_sd21) and os.path.exists(os.path.join(path_sd21, "config.json"))):
        print("[2/3] SD 2.1 VAE 다운로드 중...")
        AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").save_pretrained(path_sd21)

    # 3. SDXL
    path_sdxl = os.path.join(SAVE_DIR, "vae_sdxl")
    if not (os.path.exists(path_sdxl) and os.path.exists(os.path.join(path_sdxl, "config.json"))):
        print("[3/4] SDXL VAE 다운로드 중...")
        AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").save_pretrained(path_sdxl)

    # 4. LPIPS
    path_lpips = os.path.join(SAVE_DIR, "lpips_vgg.pth")
    if not os.path.exists(path_lpips):
        print("[3/3] LPIPS 가중치 다운로드 중...")
        lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
        torch.save(lpips_model.state_dict(), path_lpips)

    print("=== [확인 완료] 모델 준비됨 ===\n")

except Exception as e:
    print(f"\n[실패] 모델 다운로드 중 오류: {e}")

class AerobladeScanner:
    def __init__(self, model_dir, device=DEVICE):
        self.device = device
        self.vaes = {}
        
        # GPU 인식 확인
        if "cuda" in self.device:
            print(f"✅ GPU 활성화됨: [{torch.cuda.get_device_name(0)}]")
        else:
            print(f"⚠️ 경고: CPU로 실행합니다.")

        print("모델 로딩 중...")
        try:
            # 로컬 모델 로드
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd1.5"), local_files_only=True
            ).to(self.device).eval()
            
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd2.1"), local_files_only=True
            ).to(self.device).eval()

            # SDXL (없으면 경고만 하고 넘어감)
            if os.path.exists(os.path.join(model_dir, "vae_sdxl")):
                self.vaes['sdxl'] = AutoencoderKL.from_pretrained(
                    os.path.join(model_dir, "vae_sdxl"), local_files_only=True
                ).to(self.device).eval()
            
            # LPIPS 로드
            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            state_dict = torch.load(os.path.join(model_dir, "lpips_vgg.pth"), map_location=self.device)
            self.lpips_loss.load_state_dict(state_dict)
            self.lpips_loss.eval()
            
        except Exception as e:
            print(f"[오류] 모델 로딩 실패: {e}")
            exit()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def preprocess(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image).unsqueeze(0).to(self.device)
        except:
            return None

    # 디지털 워터마크/메타데이터 검사 함수
    def check_digital_traces(self, img_path):
        """
        이미지 메타데이터(Exif, PNG Info)에 'AI 생성 흔적'이 있는지 검사합니다.
        발견되면 True, 없으면 False를 반환합니다.
        """
        try:
            with Image.open(img_path) as img:
                # 1. PNG 메타데이터 확인 (가장 흔함)
                # Stable Diffusion WebUI는 'parameters'라는 키에 생성 정보를 남김
                if img.info:
                    for key in img.info.keys():
                        key_lower = key.lower()
                        val_lower = str(img.info[key]).lower()
                        
                        # 흔적 키워드 리스트
                        keywords = ['parameters', 'prompt', 'negative prompt', 'steps:']
                        if key_lower in keywords:
                            return True
                        
                        # Software 정보 확인 (예: Adobe Firefly, NovelAI)
                        if 'software' in key_lower and ('ai' in val_lower or 'stable' in val_lower):
                            return True

                # 2. (추가 가능) 다른 워터마크 패턴 검사 로직
                # 현재는 메타데이터 기반이 가장 확실하고 빠름
                
        except Exception:
            pass # 파일 읽기 오류 등은 무시하고 False 반환
            
        return False

    def get_score(self, img_tensor):
        """Min(ΔAE) 계산"""
        errors = []
        with torch.no_grad():
            for _, vae in self.vaes.items():
                recon = vae(img_tensor).sample
                loss = self.lpips_loss(img_tensor, recon).item()
                errors.append(loss)
        return min(errors) if errors else 999.0

    def scan_folder(self, target_folder, threshold):
        print(f"\n=== '{target_folder}' 폴더 검사 시작 ===")
        print(f"전략: 1차 워터마크 검사 -> (없으면) -> 2차 AEROBLADE 검사")
        
        if not os.path.exists(target_folder):
            print(f"오류: '{target_folder}' 폴더가 없습니다.")
            return

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = [f for f in os.listdir(target_folder) if f.lower().endswith(valid_ext)]
        
        if not files:
            print("폴더에 이미지 파일이 없습니다.")
            return

        results = []
        print(f"총 {len(files)}개 파일 분석 중...")

        for fname in tqdm(files):
            path = os.path.join(target_folder, fname)
            
            # [1단계] 디지털 워터마크(흔적) 검사
            has_watermark = self.check_digital_traces(path)
            
            if has_watermark:
                # 워터마크가 있으면 바로 Fake 처리 (AEROBLADE 건너뜀)
                result = "Fake (Watermark)"
                score = 0.0  # 워터마크가 있으면 오차 0으로 간주 (확실한 가짜)
            else:
                # [2단계] AEROBLADE 오차 분석
                tens = self.preprocess(path)
                if tens is None:
                    result = "Error"
                    score = 999.0
                else:
                    score = self.get_score(tens)
                    is_fake = score < threshold
                    result = "Fake (Aeroblade)" if is_fake else "Real"
            
            results.append({"파일명": fname, "판정": result, "오차점수": round(score, 5)})

        # 결과 저장
        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        
        print("\n=== 검사 완료 ===")
        print(res_df.head(10)) 
        print(f"\n전체 결과가 '{OUTPUT_FILENAME}'에 저장되었습니다.")

# ================= 실행 =================
if __name__ == "__main__":
    scanner = AerobladeScanner(LOCAL_MODEL_DIR)
    
    print(f"=== [설정] 오차 기준 점수: {THRESHOLD} ===")
    
    if os.path.exists(TARGET_IMG_FOLDER):
        scanner.scan_folder(TARGET_IMG_FOLDER, THRESHOLD)
    else:
        print(f"오류: '{TARGET_IMG_FOLDER}' 폴더를 찾을 수 없습니다.")