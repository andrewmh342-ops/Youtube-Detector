document.getElementById('scanBtn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab) {
    // content.js에 탐지 시작 명령 전송
    chrome.tabs.sendMessage(tab.id, { action: "START_DETECTION_FLOW" });
    window.close();
  }
});