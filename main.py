from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from PIL import Image
from diffusers import AutoencoderKL
import lpips
from torchvision import transforms
import io

# ================= 설정 (사용자 코드 기반) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_DIR = "local_models"  # 모델이 저장된 폴더 경로
THRESHOLD = 0.1  # 판별 기준값
# =========================================================

app = FastAPI()

# 사이트 연동을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AerobladeScanner:
    def __init__(self, model_dir, device=DEVICE):
        self.device = device
        self.vaes = {}
        
        print(f"[{device}] 로컬 에어로블레이드 모델 로딩 중...")
        try:
            # 1. SD 1.5 VAE 로드 (로컬 파일만 사용)
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd1.5"), 
                local_files_only=True
            ).to(self.device).eval()
            
            # 2. SD 2.1 VAE 로드 (로컬 파일만 사용)
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd2.1"), 
                local_files_only=True
            ).to(self.device).eval()
            
            # 3. LPIPS 로드
            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            state_dict = torch.load(os.path.join(model_dir, "lpips_vgg.pth"), map_location=self.device)
            self.lpips_loss.load_state_dict(state_dict)
            self.lpips_loss.eval()
            
            print("✅ 로컬 모델 로드 완료!")
            
        except Exception as e:
            print(f"❌ [오류] 모델 로드 실패: {e}")
            print(f"'{model_dir}' 폴더 안에 모델 파일이 있는지 확인해주세요.")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def get_score(self, img_bytes):
        """이미지 바이트를 받아 복원 오차 계산"""
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            errors = []
            with torch.no_grad():
                for _, vae in self.vaes.items():
                    recon = vae(img_tensor).sample
                    loss = self.lpips_loss(img_tensor, recon).item()
                    errors.append(loss)
            return min(errors)
        except:
            return None

# 서버 시작 시 스캐너 객체 생성
scanner = AerobladeScanner(LOCAL_MODEL_DIR)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 1. 이미지 읽기
    image_data = await file.read()
    
    # 2. 에어로블레이드 점수 계산
    score = scanner.get_score(image_data)
    
    if score is None:
        return {"error": "이미지 처리 중 오류가 발생했습니다."}

    # 3. 판별 (점수가 THRESHOLD보다 낮으면 Fake)
    is_fake = score < THRESHOLD
    result_text = "Fake (가짜)" if is_fake else "Real (진짜)"

    print(f"파일명: {file.filename} | 오차점수: {round(score, 5)} | 결과: {result_text}")

    return {
        "filename": file.filename,
        "result": result_text,
        "score": round(score, 5),
        "threshold": THRESHOLD,
        "is_fake": is_fake
    }