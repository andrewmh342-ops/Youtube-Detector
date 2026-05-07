/**
 * Aeroblade 탐지 시스템
 * Google Chrome Extension Content Script
 */

// [설정 영역]
const SERVER_URL = "http://deeptector.kro.kr";
const BASE_URL = `${SERVER_URL}/upload`;
let floatingBtn = null;
let currentTargetImg = null;

// AI 조작 이미지 탐지에 활용할 키워드 목록
const CUSTOM_AI_KEYWORDS = [
    "수정되었거나 가상으로 생성된 콘텐츠", "변경되었거나 합성된 콘텐츠",
    "Altered or Synthetic Content", "AI-generated/AI-modified content", "Altered or Synthetic Content: AI-generated"
];

function getNormalizedSiteUrl() {
    return window.location.origin.replace(/\/$/, "");
}


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

// 초기화
createFloatingButton();
checkYouTubeAITags();

(function initSiteCheck() {
    const siteUrl = getNormalizedSiteUrl();
    chrome.runtime.sendMessage({ action: "check_site", url: siteUrl }, (data) => {
        if (data && data.is_blacklisted) {
            showOverlay(
                `⚠️ 알림: 이 사이트(${siteUrl})는 AI 조작 이미지가 탐지된 기록이 있습니다. 주의하세요!`,
                "fake"
            );
            setTimeout(hideOverlay, 10000);
        }
    });
})();

// 유튜브 영상 설명란에서 AI 조작 관련 키워드 탐지
function checkYouTubeAITags() {
    console.log("[Deeptector] 유튜브 탐지 로직 실행 중...");

    if (!window.location.href.includes("youtube.com/watch")) return;

    const titleElement       = document.querySelector("h1.ytd-watch-metadata");
    const descriptionElement = document.querySelector("#description-inline-expander") ||
                               document.querySelector("#description-inner");
    const aiLabelElement     = document.querySelector('ytd-video-description-infocards-section-renderer') ||
                               document.querySelector('yt-formatted-string[path="video_info.metadata.info_row.contents"]');
    const metaDescription    = document.querySelector('meta[name="description"]')?.content || "";

    if (!titleElement && !descriptionElement && !metaDescription && !aiLabelElement) {
        console.log("로딩 중... 1.5초 뒤 재시도");
        setTimeout(checkYouTubeAITags, 1500);
        return;
    }

    const combinedText = [
        titleElement?.textContent,
        descriptionElement?.textContent,
        aiLabelElement?.textContent,
        metaDescription
    ].join(" ");

    console.log(`[Deeptector] 분석 텍스트 길이: ${combinedText.length}`);

    const detectedKeyword = CUSTOM_AI_KEYWORDS.find(keyword => {
        if (keyword.length > 5) {
            return combinedText.toLowerCase().includes(keyword.toLowerCase());
        } else {
            const regex = new RegExp(`(#|\\b)${keyword}`, 'i');
            return regex.test(combinedText);
        }
    });

    if (detectedKeyword) {
        console.log(`✅ 탐지 성공: ${detectedKeyword}`);
        showOverlay("플랫폼 공식 라벨 탐지", "fake", {
            isMetadata: true,
            keyword: detectedKeyword,
            result: "가짜 (AI 생성)",
            method: "YouTube Metadata 스캔"
        });
        setTimeout(hideOverlay, 6000);
    } else {
        console.log("[Deeptector] ❌ AI 키워드를 찾지 못했습니다.");
    }
}

// 서버로 이미지 데이터 전달 (Background 경유)
async function sendToServer(blob, filename) {
    showOverlay("🛡️ AI 모델이 분석 중입니다... <br> &nbsp;&nbsp;&nbsp;&nbsp; 잠시만 기다려주세요.", "loading");

    const blobUrl = URL.createObjectURL(blob);

    chrome.runtime.sendMessage({
        action: "upload_image",
        blobUrl: blobUrl,
        filename: filename
    }, (response) => {
        URL.revokeObjectURL(blobUrl);

        if (response && response.success) {
            const data = response.data.details[0];
            const isFake = data.result.trim().includes("가짜");

            console.log("분석 완료 - 결과:", data.result, "가짜 여부:", isFake);

            if (isFake) {
                const siteUrl = getNormalizedSiteUrl();
                console.log("가짜 판정 확인: 서버 DB에 사이트 등록을 요청합니다.");
                chrome.runtime.sendMessage({ action: "report_url", url: siteUrl });
            }

            let resultMsg = isFake
                ? `⚠️ 위험! 조작된 가짜일 확률이 높습니다.`
                : `✅ 안전! 조작되지 않은 진짜로 보입니다.`;
            showOverlay(resultMsg, isFake ? "fake" : "real", data);
            setTimeout(hideOverlay, 7000);

        } else {
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
            canvas.width  = video.videoWidth  || 1280;
            canvas.height = video.videoHeight || 720;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(async (blob) => {
                await sendToServer(blob, 'youtube_capture.jpg');
            }, 'image/jpeg', 0.9);
        };
    }
}

setInterval(injectYouTubeButton, 2000);

// 이미지 위 돋보기 노출
document.addEventListener('mouseover', (e) => {
    const target = e.target;
    if (target.tagName === 'IMG' && target.width > 100 && target.height > 100) {
        currentTargetImg = target;
        const rect = target.getBoundingClientRect();
        floatingBtn.style.top  = `${window.scrollY + rect.bottom - 60}px`;
        floatingBtn.style.left = `${window.scrollX + rect.right  - 60}px`;
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
        const mimeToExt = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif'};
        const ext = mimeToExt[blob.type] || '.jpg';
        await sendToServer(blob, `web_image${ext}`);
    } catch (e) {
        showOverlay("❌ 실패: " + e.message, "fake");
        setTimeout(hideOverlay, 3000);
    }
}

function showOverlay(text, type, data = null) {
    let overlay = document.getElementById('sg-result-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'sg-result-overlay';
        document.body.appendChild(overlay);
    }

    overlay.className = `sg-${type}`;
    overlay.style.display = 'block';

    if (type === 'loading' || !data) {
        overlay.innerHTML = `<div style="font-size: 16px; font-weight: 800; padding: 10px;">${text}</div>`;
        return;
    }

    if (data.isMetadata) {
        overlay.innerHTML = `
            <div style="font-size: 13px; color: #64748b; font-weight: 700; margin-bottom: 8px; text-align: left;">🛡️ Deeptector 보안 알림</div>
            <div style="font-size: 24px; font-weight: 800; margin-bottom: 12px; text-align: left; color: #e03131;">가짜 (AI 생성)</div>
            <div style="background: rgba(224, 49, 49, 0.1); padding: 15px; border-radius: 16px; margin-bottom: 15px; text-align: left;">
                <div style="font-size: 11px; color: #e03131; font-weight: 800; text-transform: uppercase; margin-bottom: 4px;">Detected Label</div>
                <div style="font-size: 16px; font-weight: 800; color: #e03131;">"${data.keyword}"</div>
            </div>
            <div style="font-size: 12px; line-height: 1.6; background: rgba(255,255,255,0.6); padding: 12px; border-radius: 14px; text-align: left; color: #475569;">
                💡 <strong>분석 결과:</strong> 유튜브 설명란에서 AI 제작물임을 명시하는 공식 라벨이 발견되었습니다. 물리적 분석 없이도 신뢰할 수 없는 콘텐츠로 분류됩니다.
            </div>
        `;
    } else {
        const FIRE_THR = 42.0;
        const isFireFake = data.fire_score < FIRE_THR;
        const isAeFake   = data.score < data.threshold;
        const metadataFake = data.detected_source && data.detected_source.startsWith("메타데이터 탐지");
        let displayVal, displayThr, maxVal, gaugeTitle;

        if (metadataFake) {
            displayVal = data.score; displayThr = data.threshold; maxVal = 0.15; gaugeTitle = "메타데이터 탐지";
        } else if (isFireFake && !isAeFake) {
            displayVal = data.fire_score; displayThr = FIRE_THR; maxVal = 80; gaugeTitle = "FIRE Score (주파수 분석)";
        } else {
            displayVal = data.score; displayThr = data.threshold; maxVal = 0.15; gaugeTitle = "AEROBLADE (재구성 오차)";
        }

        const thrLabel = displayThr < 1 ? displayThr.toFixed(4) : displayThr.toFixed(1);

        overlay.innerHTML = `
            <div style="font-size: 13px; color: #64748b; font-weight: 700; margin-bottom: 8px; text-align: left;">🔍 분석 완료</div>
            <div style="font-size: 24px; font-weight: 800; margin-bottom: 15px; text-align: left;">${data.result}</div>
            <div class="gauge-container">
                <div style="font-size: 11px; color: #64748b; margin-bottom: 5px; font-weight: bold; text-align: left;">📊 ${gaugeTitle}</div>
                <div class="gauge-track">
                    <div class="gauge-threshold-line" id="sg-thr-line"></div>
                    <div class="gauge-threshold-text" id="sg-thr-text">Threshold: ${thrLabel}</div>
                    <div class="gauge-point" id="sg-gauge-point"></div>
                </div>
            </div>
            <div style="font-size: 12px; line-height: 1.6; background: rgba(255,255,255,0.6); padding: 12px; border-radius: 14px; text-align: left; color: #475569;">
                💡 <strong>판단 근거:</strong> ${text.includes('\n') ? text.split('\n\n')[1] : text}
            </div>
        `;

        setTimeout(() => {
            const thrPercent   = Math.min(100, (displayThr / maxVal) * 100);
            const scorePercent = Math.min(100, (displayVal / maxVal) * 100);
            const thrLine  = document.getElementById('sg-thr-line');
            const thrText  = document.getElementById('sg-thr-text');
            const point    = document.getElementById('sg-gauge-point');
            if (thrLine)  thrLine.style.left  = `${thrPercent}%`;
            if (thrText)  thrText.style.left  = `${thrPercent}%`;
            if (point)    point.style.left    = `calc(${scorePercent}% - 8px)`;
        }, 100);
    }
}

function hideOverlay() {
    const overlay = document.getElementById('sg-result-overlay');
    if (overlay) overlay.style.display = 'none';
}

const descTarget = document.querySelector("#description");
if (descTarget) {
    const observer = new MutationObserver(() => {
        checkYouTubeAITags();
    });
    observer.observe(descTarget, { childList: true, subtree: true });
}

window.addEventListener("yt-navigate-finish", () => {
    setTimeout(checkYouTubeAITags, 1500);
    const navTarget = document.querySelector("#description") || document.querySelector("#columns");
    if (navTarget) {
        const observer = new MutationObserver(() => {
            checkYouTubeAITags();
        });
        observer.observe(navTarget, { childList: true, subtree: true });
    }
});
