// 영상 프레임 캡처 함수
async function captureVideo(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return new Promise(res => canvas.toBlob(res, 'image/jpeg'));
}

// 딱지 부착 함수
function addBadge(imgElement, score) {
    if (imgElement.parentElement.querySelector('.df-badge')) return;
    if (getComputedStyle(imgElement.parentElement).position === 'static') {
        imgElement.parentElement.style.position = 'relative';
    }
    const badge = document.createElement('div');
    badge.className = 'df-badge';
    badge.innerText = `⚠️ ${score}`;
    imgElement.parentElement.appendChild(badge);
}

// 통합 탐지 로직
async function runDetectionFlow() {
    const video = document.querySelector('video.html5-main-video') || document.querySelector('video');

    // 1. 영상 우선 탐지
    if (video && video.readyState >= 2 && !video.ended) {
        console.log("영상을 발견했습니다. 분석 중...");
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
                // 분석 완료 후 결과 오버레이로 업데이트
                showResultOverlay(res.data.result, res.data.score, "영상 분석 결과");
            } else {
                 showResultOverlay("오류", "서버 통신 실패", "오류 발생");
            }
        });
    } 
    // 2. 영상이 없을 경우 이미지 전수 조사
    else {
        console.log("영상이 없어 페이지 내 이미지를 분석합니다.");
        const images = document.querySelectorAll('img');
        let count = 0;
        images.forEach(img => {
            // 너무 작거나 외부 링크가 아닌 이미지는 제외
            if (img.src.startsWith('http') && img.width > 100 && img.height > 100) {
                count++;
                chrome.runtime.sendMessage({
                    action: "PROCESS_UPLOAD",
                    dataUrl: img.src,
                    filename: `page_image_${count}.jpg`,
                    isVideo: false
                }, (res) => {
                    // 전수 조사는 결과창 대신 '가짜'인 경우에만 딱지 부착
                    if (res && res.data && res.data.result === "가짜") {
                        addBadge(img, res.data.score);
                    }
                });
            }
        });
        if (count === 0) alert("분석할 만한 이미지가 없습니다.");
    }
}

// 분석 결과를 화면에 띄우는 함수
function showResultOverlay(result, score) {
    const existing = document.querySelector('.df-result-alert');
    if (existing) existing.remove();

    const alertDiv = document.createElement('div');
    alertDiv.className = `df-result-alert ${result === "가짜" ? "fake" : ""}`;
    
    alertDiv.innerHTML = `
        <h3>딥페이크 분석 결과</h3>
        <p>판별 : <strong>${result}</strong></p>
        <p>신뢰도 점수 : ${score}</p>
        <button id="df-close-btn">닫기</button>
    `;

    document.body.appendChild(alertDiv);

    document.getElementById('df-close-btn').addEventListener('click', () => {
        alertDiv.remove();
    });
}

// 메시지 리스너
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // 팝업의 '탐지 시작' 버튼 클릭 시
    if (msg.action === "START_DETECTION_FLOW") {
        runDetectionFlow();
    }
    
    // 이미지 우클릭 메뉴 클릭 시
    if (msg.action === "SCAN_RIGHT_CLICKED") {
        //console.log("우클릭 이미지 분석 시작...");
        //showResultOverlay("분석 중...", "서버로 전송하고 있습니다.", "이미지 분석");

        chrome.runtime.sendMessage({
            action: "PROCESS_UPLOAD",
            dataUrl: msg.url,
            filename: "context_menu.jpg",
            isVideo: false
        }, (res) => {
            if (res && res.success) {
                showResultOverlay(res.data.result, res.data.score, "이미지 분석 결과");
            } else {
                showResultOverlay("오류", "분석에 실패했습니다.", "오류 발생");
            }
        });
    }
});
// 영상 프레임을 데이터 URL(Base64)로 변환하는 함수
async function getFrameAsDataUrl(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg');
}

console.log("딥페이크 탐지 스크립트가 페이지에 로드되었습니다."); // 로드 확인용

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    console.log("메시지 수신됨:", msg.action); // 메시지 도달 확인

    if (msg.action === "RUN_GENERAL_SCAN") {
        startDetection();
        sendResponse({status: "탐지 시작됨"});
    }
    return true;
});

async function startDetection() {
    // 유튜브 메인 플레이어 비디오 태그 찾기
    const video = document.querySelector('video.html5-main-video') || document.querySelector('video');

    if (video && video.readyState >= 2) {
        console.log("비디오 캡처 중...");
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg');
        
        // background.js로 전송
        chrome.runtime.sendMessage({
            action: "UPLOAD_TO_SERVER",
            dataUrl: dataUrl,
            filename: "youtube_frame.jpg"
        });
    } else {
        console.log("유효한 비디오를 찾을 수 없습니다.");
        // 이미지 분석 로직 실행...
    }
}

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === "RUN_GENERAL_SCAN") startDetection();
});