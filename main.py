import os
import io
import torch
import lpips
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

# ================= 설정 (팀원들과 공유할 핵심 변수) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "local_models" # 모델 저장 폴더
THRESHOLD = 0.08          # 기본 판별 기준 (유튜브 캡처 위주라면 0.025로 낮추세요)
# =================================================================

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
        self.model_dir = model_dir
        self.vaes = {}
        
        # 1. 모델이 없으면 다운로드부터 진행 (기존 다운로드 코드 통합)
        self._check_and_download()
        
        # 2. 로컬 모델 로딩
        print(f"\n[{device}] 로컬 에어로블레이드 모델 로딩 중...")
        try:
            # SD 1.5 VAE 로드
            self.vaes['sd1.5'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd1.5"), local_files_only=True
            ).to(self.device).eval()
            
            # SD 2.1 VAE 로드
            self.vaes['sd2.1'] = AutoencoderKL.from_pretrained(
                os.path.join(model_dir, "vae_sd2.1"), local_files_only=True
            ).to(self.device).eval()
            
            # LPIPS 로드
            self.lpips_loss = lpips.LPIPS(net='vgg', pretrained=False).to(self.device)
            state_dict = torch.load(os.path.join(model_dir, "lpips_vgg.pth"), map_location=self.device)
            self.lpips_loss.load_state_dict(state_dict)
            self.lpips_loss.eval()
            
            print("✅ 모든 모델 로드 완료! 이제 서비스를 시작합니다.")
        except Exception as e:
            print(f"❌ [오류] 모델 로드 실패: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def _check_and_download(self):
        """모델 폴더를 확인하고 부족한 파일을 자동으로 다운로드합니다."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        print(f"=== 모델 파일 확인 중 (저장 위치: {self.model_dir}) ===")
        
        try:
            # 1. SD 1.5 VAE
            sd15_path = os.path.join(self.model_dir, "vae_sd1.5")
            if not os.path.exists(sd15_path):
                print("[1/3] SD 1.5 VAE 다운로드 중...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
                vae.save_pretrained(sd15_path)
                print("   -> 완료")

            # 2. SD 2.1 VAE
            sd21_path = os.path.join(self.model_dir, "vae_sd2.1")
            if not os.path.exists(sd21_path):
                print("[2/3] SD 2.1 VAE 다운로드 중...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
                vae.save_pretrained(sd21_path)
                print("   -> 완료")

            # 3. LPIPS 가중치
            lpips_path = os.path.join(self.model_dir, "lpips_vgg.pth")
            if not os.path.exists(lpips_path):
                print("[3/3] LPIPS 가중치 다운로드 중...")
                lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
                torch.save(lpips_model.state_dict(), lpips_path)
                print("   -> 완료")
                
        except Exception as e:
            print(f"\n❌ 다운로드 중 오류 발생: {e}\n인터넷 연결을 확인해주세요.")

    def get_score(self, img_bytes):
        """이미지 복원 오차 계산 핵심 로직"""
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

# 서버 시작 시 스캐너 객체 생성 (다운로드 로직 포함)
scanner = AerobladeScanner(SAVE_DIR)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    score = scanner.get_score(image_data)
    
    if score is None:
        return {"error": "이미지 처리 오류"}

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