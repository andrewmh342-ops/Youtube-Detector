// [설정 영역]
const SERVER_URL = "http://localhost:20408/upload";
let floatingBtn = null;
let currentTargetImg = null;

// [초기화 영역]
function createFloatingButton() {
    floatingBtn = document.createElement('div');
    floatingBtn.id = 'sg-floating-btn';
    floatingBtn.innerHTML = '🔍';
    document.body.appendChild(floatingBtn);

    floatingBtn.addEventListener('click', async () => {
        if (currentTargetImg && currentTargetImg.src) {
            floatingBtn.style.display = 'none';
            await analyzeImageUrl(currentTargetImg.src);
        }
    });
}

// 초기화 실행
createFloatingButton();

// [기능 1] 유튜브 버튼 심기
function injectYouTubeButton() {
    if (!window.location.href.includes("youtube.com/watch")) return;
    if (document.getElementById('sg-yt-button')) return;

    const btn = document.createElement('button');
    btn.id = 'sg-yt-button';
    btn.innerHTML = '🛡️ 이 영상 가짜인지 검사하기';

    const targetArea = document.querySelector('#top-row, #owner'); 
    if (targetArea) {
        targetArea.parentElement.insertBefore(btn, targetArea.nextSibling);
        btn.onclick = async () => {
            const video = document.querySelector('video');
            if (!video) { alert("영상을 찾을 수 없습니다."); return; }
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(async (blob) => {
                await sendToServer(blob, 'youtube_capture.jpg');
            }, 'image/jpeg', 0.9);
        };
    }
}
setInterval(injectYouTubeButton, 2000);

// [기능 2] 이미지 위 돋보기
document.addEventListener('mouseover', (e) => {
    const target = e.target;
    if (target.tagName === 'IMG' && target.width > 100 && target.height > 100) {
        currentTargetImg = target;
        const rect = target.getBoundingClientRect();
        floatingBtn.style.top = `${window.scrollY + rect.bottom - 60}px`;
        floatingBtn.style.left = `${window.scrollX + rect.right - 60}px`;
        floatingBtn.style.display = 'flex';
    } else if (target.id !== 'sg-floating-btn') {
        floatingBtn.style.display = 'none';
    }
});

// [공통 기능] 서버 통신 및 알림
async function analyzeImageUrl(url) {
    showOverlay("분석 중...", "loading");
    try {
        const response = await fetch(url, { mode: 'cors' }).catch(() => null);
        if (!response || !response.ok) throw new Error("보안 정책으로 이미지를 가져올 수 없습니다.");
        const blob = await response.blob();
        await sendToServer(blob, 'web_image.jpg');
    } catch (e) {
        showOverlay("❌ 실패: " + e.message, "fake");
        setTimeout(hideOverlay, 3000);
    }
}

async function sendToServer(blob, filename) {
    showOverlay("인공지능이 분석 중입니다... 잠시만 기다려주세요.", "loading");
    const formData = new FormData();
    formData.append('files', blob, filename);

    try {
        const res = await fetch(SERVER_URL, { method: 'POST', body: formData });
        const dataList = await res.json();
        const data = dataList[0];
        const isFake = data.result === "가짜";
        
        showOverlay(
            isFake ? `⚠️ 위험! 조작된 가짜일 확률이 높습니다! (점수: ${data.score})` : `✅ 안전! 조작되지 않은 진짜로 보입니다. (점수: ${data.score})`,
            isFake ? "fake" : "real"
        );
        setTimeout(hideOverlay, 5000);
    } catch (error) {
        showOverlay("❌ 서버 연결 실패!", "fake");
        setTimeout(hideOverlay, 3000);
    }
}

function showOverlay(text, type) {
    let overlay = document.getElementById('sg-result-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'sg-result-overlay';
        document.body.appendChild(overlay);
    }
    overlay.className = `sg-${type}`;
    overlay.innerText = text;
    overlay.style.display = 'block';
}

function hideOverlay() {
    const overlay = document.getElementById('sg-result-overlay');
    if (overlay) overlay.style.display = 'none';
}