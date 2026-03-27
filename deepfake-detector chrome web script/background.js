const SERVER_URL = 'http://localhost:20408/upload'; //

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

// 메시지 리스너
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "PROCESS_UPLOAD") {
        uploadToServer(request.dataUrl, request.filename)
            .then(data => sendResponse({ success: true, data: data[0] }))
            .catch(err => {
                console.error("서버 통신 실패:", err);
                sendResponse({ success: false, error: err.message });
            });
        return true; // 비동기 응답을 위해 true 반환
    }
});

// 우클릭 메뉴 설정
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "detectSpecificImage",
        title: "이 이미지 탐지하기",
        contexts: ["image"]
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "detectSpecificImage") {
        chrome.tabs.sendMessage(tab.id, { action: "SCAN_RIGHT_CLICKED", url: info.srcUrl });
    }
});