document.getElementById('scanBtn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
  // 사용자 설정값 읽기
  const maxImages = document.getElementById('maxImages').value;
  const minSize = document.getElementById('minSize').value;

  if (tab) {
    chrome.tabs.sendMessage(tab.id, { 
      action: "START_DETECTION_FLOW",
      settings: {
        maxImages: parseInt(maxImages),
        minSize: parseInt(minSize)
      }
    });
    window.close();
  }
});