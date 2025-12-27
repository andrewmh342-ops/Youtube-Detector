import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import lpips
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ================= 설정 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_DIR = "local_models" # 다운로드한 모델 폴더 경로

# 경로 설정
TRAIN_IMG_FOLDER = "train_pic"   # (수동 설정 시 사용 안 함)
TRAIN_CSV_PATH = "train.csv"     # (수동 설정 시 사용 안 함)
TARGET_IMG_FOLDER = "pic"        # [중요] 판별할 대상 폴더
OUTPUT_FILENAME = "result.csv"   # 결과 저장 파일명
# ========================================

class AerobladeScanner:
    def __init__(self, model_dir, device=DEVICE):
        self.device = device
        self.vaes = {}
        
        print(f"[{device}] 모델 로딩 중...")
        try:
            # 로컬 모델 로드 (인터넷 없이)
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd1.5"), local_files_only=True
            ).to(self.device).eval()
            
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd2.1"), local_files_only=True
            ).to(self.device).eval()
            
            # LPIPS 로드
            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            state_dict = torch.load(os.path.join(model_dir, "lpips_vgg.pth"), map_location=self.device)
            self.lpips_loss.load_state_dict(state_dict)
            self.lpips_loss.eval()
            
        except Exception as e:
            print(f"[오류] 모델을 불러올 수 없습니다: {e}")
            print("먼저 download_models.py를 실행해 모델을 다운로드했는지 확인하세요.")
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

    def get_score(self, img_tensor):
        """논문의 Min(ΔAE) 계산: 여러 모델 중 최소 오차 반환"""
        errors = []
        with torch.no_grad():
            for _, vae in self.vaes.items():
                recon = vae(img_tensor).sample
                loss = self.lpips_loss(img_tensor, recon).item()
                errors.append(loss)
        return min(errors) if errors else 999.0

    def calibrate(self, train_csv, train_folder):
        """(수동 설정 시 사용되지 않음)"""
        print("\n=== 1단계: 기준값(Threshold) 설정 (Calibration) ===")
        # ... (생략 가능하지만 코드 보존) ...
        return 0.04 # 기본값 리턴

    def scan_folder(self, target_folder, threshold):
        """대상 폴더의 모든 이미지를 스캔"""
        print(f"\n=== 2단계: '{target_folder}' 폴더 검사 시작 ===")
        
        # 폴더 내 이미지 파일 리스트업
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = [f for f in os.listdir(target_folder) if f.lower().endswith(valid_ext)]
        
        if not files:
            print(f"폴더 '{target_folder}'에 이미지 파일이 없습니다.")
            return

        results = []
        print(f"총 {len(files)}개 파일 분석 중...")

        for fname in tqdm(files):
            path = os.path.join(target_folder, fname)
            tens = self.preprocess(path)
            
            if tens is None:
                result = "Error"
                score = 0.0
            else:
                score = self.get_score(tens)
                # [판별 로직]
                # 점수 < 0.1 이면 Fake(가짜)
                # 점수 >= 0.1 이면 Real(진짜)
                is_fake = score < threshold
                result = "Fake (가짜)" if is_fake else "Real (진짜)"
            
            results.append({"파일명": fname, "판정": result, "오차점수": round(score, 5)})

        # 결과 저장
        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')
        
        print("\n=== 검사 완료 ===")
        print(res_df.head(10)) # 상위 10개 출력
        print(f"\n전체 결과가 '{OUTPUT_FILENAME}'에 저장되었습니다.")

# ================= 실행 =================
if __name__ == "__main__":
    scanner = AerobladeScanner(LOCAL_MODEL_DIR)
    
    # [수정됨] 학습 데이터 계산을 건너뛰고 0.1로 고정
    # threshold = scanner.calibrate(TRAIN_CSV_PATH, TRAIN_IMG_FOLDER)
    threshold = 0.1
    
    print(f"=== [수동 설정] 오차 기준 점수를 {threshold}로 고정합니다 ===")
    
    # 2. 실제 pic 폴더 검사하기
    if os.path.exists(TARGET_IMG_FOLDER):
        scanner.scan_folder(TARGET_IMG_FOLDER, threshold)
    else:
        print(f"오류: '{TARGET_IMG_FOLDER}' 폴더를 찾을 수 없습니다.")