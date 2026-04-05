/**
 * Aeroblade 탐지 시스템
 * Google Chrome Extension Content Script
 */
// [설정 영역]
const SERVER_URL = "http://deeptector.kro.kr";
const BASE_URL = `${SERVER_URL}/upload`;
let floatingBtn = null;
let currentTargetImg = null;

// AI 조작 이미지 탐지에 활용할 키워드 목록 (유튜브 라벨, 유명 AI 모델, 제작 도구, 제작자 관행 등)
const CUSTOM_AI_KEYWORDS = [
    "AI", "aiart", "aiasmr", "deepfake", "sora", "Altered or synthetic content", "수정되었거나 가상으로 생성된 콘텐츠", 
    "AI 제작", "변경되었거나 합성된 콘텐츠", "AI 기술 활용", "Sora", "Midjourney", "Stable Diffusion", "DALL-E", 
    "Runway Gen", "Pika Labs", "Luma Dream Machine", "Flux.1", "Deepfake", "AI Generated", "AI-assisted", 
    "Generative AI", "인공지능 생성", "합성 영상", "AI 페이스", "Face Swap","Made with AI", "Prompt by", "AI-powered", "Created using AI"
];


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

// 초기화 영역
checkCurrentSite();
createFloatingButton(); //
checkYouTubeAITags();

// 페이지 접속 시 위험 사이트 확인
async function checkCurrentSite() {
    const currentFullUrl = window.location.href; 
    chrome.runtime.sendMessage({ action: "check_site", url: currentFullUrl }, (data) => {
        if (data && data.is_blacklisted) {
            showOverlay(`⚠️ 알림: 이 페이지(${currentFullUrl})는 가짜 이미지가 탐지된 기록이 있습니다!`, "fake");
            setTimeout(hideOverlay, 10000);
        }
    });
}
// [자동 실행] 페이지 로드 즉시 서버에 해당 사이트가 DB에 있는지 확인
(function init() {
    const currentSite = window.location.origin;
    chrome.runtime.sendMessage({ action: "check_site", url: currentSite }, (data) => {
        if (data && data.is_blacklisted) {
            // DB에 등록된 사이트라면 알림 표시 (10초간 노출)
            showOverlay(`⚠️ 알림: 이 사이트(${currentSite})는 AI 조작 이미지가 탐지된 기록이 있습니다. 주의하세요!`, "fake");
            // 자동 알림이므로 10초간 충분히 노출
            setTimeout(hideOverlay, 10000);
        }
    });
})();

// 유튜브 영상 설명란에서 AI 조작 관련 키워드 탐지
function checkYouTubeAITags() {
    console.log("[Deeptector] 유튜브 탐지 로직 실행 중...");

    if (!window.location.href.includes("youtube.com/watch")) return;

    const titleElement = document.querySelector("h1.ytd-watch-metadata");
    const descriptionElement = document.querySelector("#description-inline-expander") || 
                               document.querySelector("#description-inner");
    const metaDescription = document.querySelector('meta[name="description"]')?.content || "";

    if (!titleElement && !descriptionElement && !metaDescription) {
        console.log("로딩 중... 1.5초 뒤 재시도");
        setTimeout(checkYouTubeAITags, 1500);
        return;
    }
    const combinedText = (titleElement?.textContent || "") + " " + 
                         (descriptionElement?.textContent || "") + " " + 
                         metaDescription;

    const detectedKeyword = CUSTOM_AI_KEYWORDS.find(keyword => {
        const regex = new RegExp(`(#|\\b)${keyword}`, 'i');
        return regex.test(combinedText);
    });

    if (detectedKeyword) {
        console.log(`✅ 탐지 성공: ${detectedKeyword}`);
        showOverlay(`⚠️ 유튜브 영상 태그에 AI 키워드가 감지되었습니다! \n ▶ 감지된 키워드: ${detectedKeyword}`, "fake");
        if (typeof hideOverlay === 'function') setTimeout(hideOverlay, 5000);
    } else {
        console.log("[Deeptector] ❌ AI 키워드를 찾지 못했습니다.");
    }
}

// 서버로 이미지 데이터 전달 (Background 경유)
async function sendToServer(blob, filename) {
    // 분석 시작 알림 표시
    showOverlay("🛡️ AI 모델이 분석 중입니다... 잠시만 기다려주세요.", "loading");

    // Blob 데이터를 Background Service Worker로 전달하기 위한 임시 URL 생성
    const blobUrl = URL.createObjectURL(blob);

    // Background Script를 background.js에 메시지 전송
    chrome.runtime.sendMessage({
        action: "upload_image",
        blobUrl: blobUrl,
        filename: filename
    }, (response) => {
        // 임시 URL 메모리 해제
        URL.revokeObjectURL(blobUrl);

        if (response && response.success) {
            // 서버 응답 구조에 맞게 데이터 추출
            const data = response.data.details[0]; 
            const isFake = data.result.trim().includes("가짜");
            
            console.log("분석 완료 - 결과:", data.result, "가짜 여부:", isFake);

            // 가짜로 판명된 경우 자동으로 현재 사이트 URL 리포트
            if (isFake) {
                const currentDomain = window.location.href;
                console.log("가짜 판정 확인: 서버 DB에 사이트 등록을 요청합니다.");
                chrome.runtime.sendMessage({ 
                    action: "report_url", 
                    url: currentDomain // 도메인 주소 전송
                });
            }

            // 결과 화면 출력 (점수 및 기준값 포함)
            const score = data.score.toFixed(4);
            const threshold = data.threshold ? data.threshold.toFixed(4) : "0.0500";
            const method = data.method || "AEROBLADE Ensemble";

            let resultMsg = "";
            if (isFake) {
                resultMsg = `⚠️ 위험! 조작된 가짜(AI 생성)일 확률이 높습니다!\n\n▶ 분석 방식: ${method}\n▶ 측정 점수: ${score} (기준값 ${threshold} 미만)`;
            } else {
                resultMsg = `✅ 안전! 조작되지 않은 진짜로 보입니다.\n\n▶ 분석 방식: ${method}\n▶ 측정 점수: ${score} (기준값 ${threshold} 이상)`;
            }

            showOverlay(resultMsg, isFake ? "fake" : "real");
            
            // 정보가 많으므로 7초간 충분히 노출
            setTimeout(hideOverlay, 7000);

        } else {
            // 에러 처리: 서버가 꺼져있거나 네트워크 문제인 경우
            console.error("서버 통신 에러:", response ? response.error : "Unknown Error");
            showOverlay("❌ 서버 연결 실패! 백엔드 서버가 실행 중인지 확인하세요.", "fake");
            setTimeout(hideOverlay, 3000);
        }
    });
}

// 유튜브 버튼 심기
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

// 이미지 위 돋보기 노출
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

// 서버 통신 및 알림
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

const observer = new MutationObserver((mutations) => {
    detectYouTubeAITag();
});

const target = document.querySelector("#description");
if (target) {
    observer.observe(target, { childList: true, subtree: true });
}

window.addEventListener("yt-navigate-finish", () => {
    setTimeout(checkYouTubeAITags, 1500); 
    // 설명란 영역이 동적으로 변할 때를 위해 감시 시작
    const target = document.querySelector("#description") || document.querySelector("#columns");
    if (target) {
        const observer = new MutationObserver(() => {
            checkYouTubeAITags();
        });
        observer.observe(target, { childList: true, subtree: true });
    }
});