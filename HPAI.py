import os
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------
# [설정] 이미지 폴더 이름
IMAGE_FOLDER = "images"
# 검사할 파일 확장자
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')

# SuSy가 사용하는 6개 클래스 이름 (모델 카드 / 예제 기준)
# authentic = 실제 사진, 나머지는 각 생성 모델
CLASS_NAMES = [
    "authentic",
    "dalle-3-images",
    "diffusiondb",
    "midjourney-images",
    "midjourney_tti",
    "realisticSDXL",
]
# ---------------------------------------------------------


def load_susy_model(device="cpu"):
    """
    HPAI-BSC/SuSy에서 TorchScript 모델(SuSy.pt)을 받아와 로드
    """
    print("=" * 60)
    print("🚀 [HPAI-BSC/SuSy] 모델 로딩 중...")
    print("=" * 60)

    # HF에서 SuSy.pt 다운로드 (처음 한 번은 인터넷 필요)
    model_path = hf_hub_download(
        repo_id="HPAI-BSC/SuSy",
        filename="SuSy.pt",
    )

    # TorchScript 모델 로드
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    print("✅ SuSy 모델 로드 완료!\n")
    return model


# SuSy 입력용 전처리 : 224x224, [0,1] 범위 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),            # 자동으로 /255 해줌 (0~1 float)
])


def classify_image(model, image_path, device="cpu"):
    """
    한 장의 이미지를 SuSy로 분류하고
    (label 문자열, 확률(float 0~1)) 반환
    """
    # 이미지 열기 (RGB 강제)
    img = Image.open(image_path).convert("RGB")

    # 전처리 + 배치 차원 추가 [1,3,224,224]
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)                # [1, 6]
        probs = torch.softmax(logits, dim=1)[0]  # [6]

    top_idx = int(torch.argmax(probs))
    top_prob = float(probs[top_idx])
    label = CLASS_NAMES[top_idx]

    return label, top_prob


def run_susy():
    # 0. 디바이스 선택 (GPU 있으면 cuda 사용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 사용 디바이스: {device}")

    # 1. 폴더 확인
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 오류: '{IMAGE_FOLDER}' 폴더가 없습니다. 현재 위치에 폴더를 만들어주세요.")
        return

    # 2. 모델 로드
    try:
        model = load_susy_model(device=device)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("아래 패키지가 설치되어 있는지 확인하세요:")
        print("  pip install torch torchvision huggingface_hub pillow")
        return

    # 3. 파일 목록 가져오기
    files = [f for f in os.listdir(IMAGE_FOLDER)
             if f.lower().endswith(VALID_EXTENSIONS)]

    if not files:
        print(f"⚠️ '{IMAGE_FOLDER}' 폴더 안에 이미지 파일이 없습니다.")
        return

    # 출력 포맷 설정
    print(f"{'파일명':<25} | {'판정 (Label)':<30} | {'확률 (Score)':<10}")
    print("-" * 80)

    ai_count = 0
    real_count = 0

    # 4. 분석 시작
    for file_name in files:
        file_path = os.path.join(IMAGE_FOLDER, file_name)

        try:
            label, prob = classify_image(model, file_path, device=device)

            # authentic = 실제, 나머지 = 생성 이미지로 해석
            if label == "authentic":
                display_label = "📷 Real (authentic)"
                real_count += 1
            else:
                display_label = f"🤖 AI ({label})"
                ai_count += 1

            print(f"{file_name:<25} | {display_label:<30} | {prob*100:6.2f}%")

        except Exception as e:
            print(f"{file_name:<25} | ❌ 파일 에러 ({e})")

    # 5. 최종 요약
    print("-" * 80)
    print("📊 [최종 요약]")
    print(f"총 검사 파일 : {len(files)}개")
    print(f"🤖 AI(합성)    : {ai_count}개")
    print(f"📷 실제(authentic): {real_count}개")
    print("-" * 80)


if __name__ == "__main__":
    run_susy()
