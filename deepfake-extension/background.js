// background.js
const SERVER_BASE = "http://deeptector.kro.kr";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // 1. 사이트 접속 시 DB 조회
    if (request.action === "check_site") {
        const cleanUrl = request.url.replace(/\/$/, ""); 
        fetch(`${SERVER_BASE}/check-site?url=${encodeURIComponent(cleanUrl)}`)
            .then(res => res.json())
            .then(data => sendResponse(data))
            .catch(() => sendResponse({ is_blacklisted: false }));
        return true;
    }

    
    // 2. 가짜 판정 시 DB 등록
    if (request.action === "report_url") {
        const cleanUrl = request.url.trim().replace(/\/$/, ""); 
        console.log("DB 등록 요청 시작:", cleanUrl);

        const params = new URLSearchParams();
        params.append('url', cleanUrl);

        fetch(`${SERVER_BASE}/report-fake-url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: params
        })
        .then(res => res.json())
        .then(data => console.log("서버 DB 등록 성공 응답:", data))
        .catch(err => console.error("서버 DB 등록 실패:", err));
        
        return true; // 비동기 응답 유지
    }

    // 3. 이미지 업로드 탐지
    if (request.action === "upload_image") {
        fetch(request.blobUrl).then(res => res.blob()).then(blob => {
            const formData = new FormData();
            formData.append('files', blob, request.filename);
            formData.append('expected_answer', 'none');

            fetch(`${SERVER_BASE}/upload`, { method: 'POST', body: formData })
                .then(res => res.json())
                .then(data => sendResponse({ success: true, data: data }))
                .catch(err => sendResponse({ success: false, error: err.message }));
        });
        return true;
    }
});