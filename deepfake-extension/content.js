/**
 * Aeroblade 탐지 시스템
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
    // 기본적인 플로팅 버튼 스타일
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

// 초기화 실행
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
// 유튜브 페이지 이동 시 대응을 위해 반복 실행
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
    showOverlay("🛡️ AI 모델이 분석 중입니다... 잠시만 기다려주세요.", "loading");
    const formData = new FormData();
    formData.append('files', blob, filename);
    formData.append('expected_answer', 'none'); // [수정] 서버에서 요구하는 필수 필드 추가

    try {
        const res = await fetch(SERVER_URL, { method: 'POST', body: formData });
        const responseData = await res.json();
        
        // [핵심 수정] 서버 응답이 { "summary": ..., "details": [...] } 형태이므로 구조에 맞게 수정
        const data = responseData.details[0]; 
        
        const isFake = data.result === "가짜";
        
        // 서버에서 보내는 데이터를 화면에 맞게 가공
        const score = data.score.toFixed(4); // LPIPS 오차 점수
        const threshold = data.threshold_used.toFixed(4); // 적용된 기준값
        const method = data.method; // 분석 방식
        
        let resultMsg = "";
        if (isFake) {
            resultMsg = `⚠️ 위험! 조작된 가짜(AI 생성)일 확률이 높습니다!\n\n▶ 분석 방식: ${method}\n▶ 측정 점수: ${score} (기준값 ${threshold} 미만)`;
        } else {
            resultMsg = `✅ 안전! 조작되지 않은 진짜로 보입니다.\n\n▶ 분석 방식: ${method}\n▶ 측정 점수: ${score} (기준값 ${threshold} 이상)`;
        }

        showOverlay(resultMsg, isFake ? "fake" : "real");
        
        // 정보가 많으므로 충분히 읽을 수 있게 7초간 표시
        setTimeout(hideOverlay, 7000);
    } catch (error) {
        showOverlay("❌ 서버 연결 실패! 백엔드 서버가 실행 중인지 확인하세요.", "fake");
        setTimeout(hideOverlay, 3000);
    }
}

function showOverlay(text, type) {
    let overlay = document.getElementById('sg-result-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'sg-result-overlay';
        overlay.style.cssText = `
            position: fixed; top: 30px; right: 30px; z-index: 999999;
            padding: 20px; border-radius: 12px; font-weight: bold;
            white-space: pre-line; box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            min-width: 320px; line-height: 1.6; pointer-events: none;
            font-size: 15px; border: 2px solid rgba(0,0,0,0.1);
            font-family: 'Malgun Gothic', sans-serif;
        `;
        document.body.appendChild(overlay);
    }
    
    // 타입별 색상 테마 설정
    const themes = {
        loading: { bg: '#fff9db', text: '#856404', border: '#ffeeba' },
        fake: { bg: '#fff5f5', text: '#c92a2a', border: '#ffa8a8' },
        real: { bg: '#ebfbee', text: '#2b8a3e', border: '#b2f2bb' }
    };
    
    const theme = themes[type] || themes.loading;
    overlay.style.backgroundColor = theme.bg;
    overlay.style.color = theme.text;
    overlay.style.borderColor = theme.border;
    overlay.innerText = text; // 텍스트만 출력
    overlay.style.display = 'block';
}

function hideOverlay() {
    const overlay = document.getElementById('sg-result-overlay');
    if (overlay) overlay.style.display = 'none';
}