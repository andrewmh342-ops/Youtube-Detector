import os
import torch
import lpips
from diffusers import AutoencoderKL

# 저장할 폴더 이름
SAVE_DIR = "local_models"

# 폴더가 없으면 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"=== 모델 다운로드를 시작합니다 (저장 위치: {SAVE_DIR}) ===")
print("인터넷 연결 상태를 확인해주세요...\n")

try:
    # 1. Stable Diffusion 1.5 VAE
    print("[1/3] SD 1.5 VAE 다운로드 중...")
    vae_sd15 = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae_sd15.save_pretrained(os.path.join(SAVE_DIR, "vae_sd1.5"))
    print("  -> 완료")

    # 2. Stable Diffusion 2.1 VAE
    print("[2/3] SD 2.1 VAE 다운로드 중...")
    vae_sd21 = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae_sd21.save_pretrained(os.path.join(SAVE_DIR, "vae_sd2.1"))
    print("  -> 완료")

    # 3. LPIPS (VGG) 가중치
    print("[3/3] LPIPS 가중치 다운로드 중...")
    # net='vgg'로 초기화하면 자동으로 vgg16 가중치를 받아옵니다.
    lpips_model = lpips.LPIPS(net='vgg', pretrained=True)
    torch.save(lpips_model.state_dict(), os.path.join(SAVE_DIR, "lpips_vgg.pth"))
    print("  -> 완료")

    print("\n=== [성공] 모든 모델이 저장되었습니다! ===")
    print(f"이제 인터넷을 끊고 스캐너를 실행할 수 있습니다.")

except Exception as e:
    print(f"\n[실패] 다운로드 중 오류가 발생했습니다: {e}")
    print("인터넷 연결이나 방화벽 설정을 확인해주세요.")