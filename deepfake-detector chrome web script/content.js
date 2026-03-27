// [1] 딱지 부착 함수
function addBadge(imgElement, score) {
    if (imgElement.parentElement.querySelector('.df-badge')) return;
    if (getComputedStyle(imgElement.parentElement).position === 'static') {
        imgElement.parentElement.style.position = 'relative';
    }
    const badge = document.createElement('div');
    badge.className = 'df-badge';
    badge.innerText = `⚠️ 딥페이크 ${score}`;
    imgElement.parentElement.appendChild(badge);
}

// [2] 결과 오버레이 창 표시 함수
function showResultOverlay(result, score, typeText = "분석 결과") {
    const existing = document.querySelector('.df-result-alert');
    if (existing) existing.remove();
    const alertDiv = document.createElement('div');
    alertDiv.className = `df-result-alert ${result === "가짜" ? "fake" : ""}`;
    alertDiv.innerHTML = `
        <h3>${typeText}</h3>
        <p>판별 : <strong>${result}</strong></p>
        <p>신뢰도 점수 : ${score}</p>
        <button id="df-close-btn">닫기</button>`;
    document.body.appendChild(alertDiv);
    document.getElementById('df-close-btn').onclick = () => alertDiv.remove();
}

// [3] 통합 탐지 실행 로직
async function runDetectionFlow(settings) {
    // 유튜브 메인 비디오 또는 일반 비디오 태그 찾기
    const video = document.querySelector('video.html5-main-video') || document.querySelector('video');

    if (video && video.readyState >= 2 && !video.ended) {
        console.log("영상 분석을 시작합니다...");
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg');

        chrome.runtime.sendMessage({
            action: "PROCESS_UPLOAD",
            dataUrl: dataUrl,
            filename: "video_frame.jpg",
            isVideo: true
        }, (res) => {
            if (res && res.success) {
                showResultOverlay(res.data.result, res.data.score, "영상 분석 결과");
            } else {
                showResultOverlay("오류", "서버 통신 실패", "오류 발생");
            }
        });
    } else {
        const images = Array.from(document.querySelectorAll('img'));
        const { maxImages, minSize } = settings;
        let processedCount = 0;

        console.log(`탐지 시작: 최소 크기 ${minSize}px, 최대 ${maxImages}개`);

        for (let img of images) {
            // 최대 개수 도달 시 중단
            if (processedCount >= maxImages) break;
            // 사이즈 미달 및 외부 주소가 아닌 이미지 건너뛰기
            if (img.src.startsWith('http') && img.naturalWidth >= minSize && img.naturalHeight >= minSize) {
                processedCount++;
                chrome.runtime.sendMessage({
                    action: "PROCESS_UPLOAD",
                    dataUrl: img.src,
                    filename: `img_${processedCount}.jpg`,
                    isVideo: false
                }, (res) => {
                    if (res && res.data && res.data.result === "가짜") {
                        addBadge(img, res.data.score);
                    }
                });
            }
        }
        if (processedCount === 0) alert(`${minSize}px 이상의 이미지를 찾지 못했습니다.`);
    }
}

// [4] 단일 메시지 리스너
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    console.log("메시지 수신됨:", msg.action);
    if (msg.action === "START_DETECTION_FLOW") {
        runDetectionFlow(msg.settings);
        sendResponse({status: "탐지 프로세스 시작"});
    }
    if (msg.action === "SCAN_RIGHT_CLICKED") {
        chrome.runtime.sendMessage({
            action: "PROCESS_UPLOAD",
            dataUrl: msg.url,
            filename: "context_menu.jpg",
            isVideo: false
        }, (res) => {
            if (res && res.success) {
                showResultOverlay(res.data.result, res.data.score, "이미지 분석 결과");
            }
        });
    }
    return true;
});

console.log("Deepfake Detector 스크립트 로드 완료");