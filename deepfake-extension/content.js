/**
 * Aeroblade 앙상블 탐지 시스템
 * Google Chrome Extension Content Script
 */

// [설정 영역]
const SERVER_URL = "http://localhost:20408/upload";
let floatingBtn = null;
let currentTargetImg = null;

// [초기화 영역]
function createFloatingButton() {
    floatingBtn = document.createElement('div');
    floatingBtn.id = 'sg-floating-btn';
    floatingBtn.innerHTML = '🔍';
    floatingBtn.style.cssText = `
        position: absolute; width: 40px; height: 40px; 
        background: #007bff; color: white; border-radius: 50%;
        display: none; align-items: center; justify-content: center;
        cursor: pointer; z-index: 999998; font-size: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3); transition: transform 0.2s;
    `;
    document.body.appendChild(floatingBtn);

    floatingBtn.addEventListener('click', async () => {
        if (currentTargetImg && currentTargetImg.src) {
            floatingBtn.style.display = 'none';
            await analyzeImageUrl(currentTargetImg.src);
        }
    });
}

createFloatingButton();

// [기능 1] 유튜브 버튼 심기
function injectYouTubeButton() {
    if (!window.location.href.includes("youtube.com/watch")) return;
    if (document.getElementById('sg-yt-button')) return;

    const btn = document.createElement('button');
    btn.id = 'sg-yt-button';
    btn.innerHTML = '🛡️ 이 영상 가짜인지 검사하기';
    btn.style.cssText = `
        background: #065fd4; color: white; border: none; 
        padding: 10px 15px; border-radius: 18px; font-weight: bold;
        cursor: pointer; margin-left: 10px; font-size: 14px;
    `;

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

// [기능 2] 이미지 위 돋보기 노출
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
    showOverlay("🖼️ 이미지 데이터를 가져오는 중...", "loading");
    try {
        const response = await fetch(url, { mode: 'cors' }).catch(() => null);
        if (!response || !response.ok) throw new Error("보안 정책(CORS)으로 이미지를 가져올 수 없습니다.");
        const blob = await response.blob();
        await sendToServer(blob, 'web_image.jpg');
    } catch (e) {
        showOverlay("❌ 실패: " + e.message, "fake");
        setTimeout(hideOverlay, 3000);
    }
}

async function sendToServer(blob, filename) {
    showOverlay("🛡️ 앙상블 모델이 분석 중입니다... (Aeroblade + EfficientNet)", "loading");
    const formData = new FormData();
    formData.append('files', blob, filename);

    try {
        const res = await fetch(SERVER_URL, { method: 'POST', body: formData });
        const dataList = await res.json();
        const data = dataList[0]; 
        
        const isFake = data.result === "가짜";
        
        // 서버에서 받아온 확률값 처리 (값이 없을 경우 0으로 처리하여 NaN 방지)
        const aeroProb = ((data.aero_prob || 0) * 100).toFixed(1);
        const auxProb = ((data.aux_prob || 0) * 100).toFixed(1);
        const finalProb = ((data.final_score || 0) * 100).toFixed(1);

        let resultMsg = isFake ? `⚠️ 위조 판정: 이 데이터는 [가짜]일 가능성이 높습니다!\n\n` : `✅ 안전 판정: 이 데이터는 [진짜]일 가능성이 높습니다.\n\n`;
        
        resultMsg += `▶ 최종 가짜 확률(위험도): ${finalProb}%\n`;
        resultMsg += `(가짜 확률이 50% 미만이면 '진짜'로 판정합니다)\n`;
        resultMsg += `────────────────\n`;
        resultMsg += `- 에어로 기여도: ${aeroProb}%\n`;
        resultMsg += `- 보조모델 기여도: ${auxProb}%\n`;
        resultMsg += `────────────────\n`;
        resultMsg += `▶ 분석 방식: ${data.method || "앙상블 분석"}`;

        showOverlay(resultMsg, isFake ? "fake" : "real");
        setTimeout(hideOverlay, 10000); // 정보가 많으므로 10초간 표시
    } catch (error) {
        showOverlay("❌ 서버 연결 실패! 백엔드 서버가 실행 중인지 확인하세요.", "fake");
        setTimeout(hideOverlay, 3000);
    }
}

// [UI 영역] 결과 오버레이
function showOverlay(text, type) {
    let overlay = document.getElementById('sg-result-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'sg-result-overlay';
        overlay.style.cssText = `
            position: fixed; top: 30px; right: 30px; z-index: 999999;
            padding: 20px; border-radius: 12px; font-weight: bold;
            white-space: pre-line; box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            min-width: 340px; line-height: 1.6; pointer-events: none;
            font-size: 15px; border: 2px solid rgba(0,0,0,0.1);
            font-family: 'Malgun Gothic', sans-serif;
            max-height: 85vh; overflow-y: auto;
        `;
        document.body.appendChild(overlay);
    }
    
    const themes = {
        loading: { bg: '#fff9db', text: '#856404', border: '#ffeeba' },
        fake: { bg: '#fff5f5', text: '#c92a2a', border: '#ffa8a8' },
        real: { bg: '#ebfbee', text: '#2b8a3e', border: '#b2f2bb' }
    };
    
    const theme = themes[type] || themes.loading;
    overlay.style.backgroundColor = theme.bg;
    overlay.style.color = theme.text;
    overlay.style.borderColor = theme.border;
    overlay.innerText = text;
    overlay.style.display = 'block';
}

function hideOverlay() {
    const overlay = document.getElementById('sg-result-overlay');
    if (overlay) overlay.style.display = 'none';
}