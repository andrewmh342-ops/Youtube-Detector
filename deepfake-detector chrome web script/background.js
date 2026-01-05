// 서버 주소 설정
const SERVER_URL = 'http://localhost:20408/upload';

// 서버 업로드 공통 함수
async function uploadToServer(dataUrl, filename) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const formData = new FormData();
    formData.append('files', blob, filename);

    const response = await fetch(SERVER_URL, {
        method: 'POST',
        body: formData
    });
    return await response.json();
}

// 통합 메시지 리스너
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "PROCESS_UPLOAD") {
        uploadToServer(request.dataUrl, request.filename)
            .then(data => {
                const result = data[0];
                // 영상 프레임 분석일 경우 알림 띄우기
                if (request.isVideo) {
                    chrome.notifications.create({
                        type: 'basic',
                        iconUrl: 'icon.png',
                        title: '딥페이크 영상 분석 결과',
                        message: `판별: ${result.result} (점수: ${result.score})`,
                        priority: 2
                    });
                }
                sendResponse({ success: true, data: result });
            })
            .catch(err => sendResponse({ success: false, error: err.message }));
        return true; 
    }
});

// 우클릭 메뉴 클릭 시 처리
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "detectSpecificImage") {
        chrome.tabs.sendMessage(tab.id, { action: "SCAN_RIGHT_CLICKED", url: info.srcUrl });
    }
});

// 메시지 리스너: Content Script로부터 받은 이미지를 서버로 전송
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "UPLOAD_TO_SERVER") {
        // Base64 데이터를 Blob으로 변환
        fetch(request.dataUrl)
            .then(res => res.blob())
            .then(blob => {
                const formData = new FormData();
                formData.append('files', blob, request.filename); // 서버 필드명 'files'

                return fetch(SERVER_URL, {
                    method: 'POST',
                    body: formData
                });
            })
            .then(res => res.json())
            .then(data => {
                const result = data[0];
                // 알림 띄우기
                chrome.notifications.create({
                    type: 'basic',
                    iconUrl: 'icon.png',
                    title: '딥페이크 분석 결과',
                    message: `결과: ${result.result} (점수: ${result.score})`,
                    priority: 2
                });
                sendResponse({ success: true, data: result });
            })
            .catch(err => {
                console.error("서버 통신 실패:", err);
                sendResponse({ success: false, error: err.message });
            });
        return true; // 비동기 응답 유지
    }
});

// 우클릭 메뉴 생성
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "detectSpecificImage",
        title: "이 이미지 탐지하기",
        contexts: ["image"]
    });
});

// 서버 분석 요청 공통 함수
async function requestAnalysis(blob, filename) {
    const formData = new FormData();
    formData.append('files', blob, filename); // 서버 필드명 'files'에 맞춤

    const response = await fetch('http://localhost:20408/upload', {
        method: 'POST',
        body: formData
    });
    const results = await response.json();
    return results[0]; // 분석 결과 반환
}

// 메시지 리스너 (컨텐츠 스크립트와의 통신)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ANALYZE_DATA") {
        fetch(request.url)
            .then(res => res.blob())
            .then(blob => requestAnalysis(blob, request.filename))
            .then(result => sendResponse(result))
            .catch(err => console.error("분석 에러:", err));
        return true; // 비동기 응답 처리
    }
    
    if (request.action === "SHOW_NOTIFY") {
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icon.png',
            title: '딥페이크 영상 분석 결과',
            message: `판별: ${request.result} (점수: ${request.score})`,
            priority: 2
        });
    }
});

// 우클릭 메뉴 클릭 시 처리
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "detectSpecificImage") {
        chrome.tabs.sendMessage(tab.id, { 
            action: "SCAN_SPECIFIC_IMAGE", 
            url: info.srcUrl 
        });
    }
});