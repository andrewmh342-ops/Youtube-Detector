 Aeroblade Ensemble: Real-time Deepfake Detection & Defense Network

 Aeroblade Original source code (https://github.com/jonasricker/aeroblade)

A comprehensive deepfake detection system featuring a FastAPI backend for ensemble analysis and a Chrome Extension for real-time monitoring and collective intelligence-based URL blacklisting.

 Overview
This project aims to combat the spread of AI-generated misinformation by providing users with immediate feedback on the authenticity of web media. Unlike traditional detectors, this system employs an Ensemble Detection approach—combining visual artifacts, frequency analysis, and metadata verification—and shares threat intelligence across a decentralized user network.

 Key Features
Ensemble AI Analysis: Utilizes AEROBLADE (VAE Reconstruction Error) and FIRE (Frequency Integrated Reconstruction Error) to detect subtle AI fingerprints.

Multi-Platform Metadata Verification: Automatically inspects digital signatures from Gemini (Google SynthID/IPTC), ChatGPT (DALL-E 3/C2PA), and Grok (Flux/PNG Info).

Real-time YouTube Integration: Adds a "Shield" button directly to the YouTube interface to analyze video frames instantly.

Collective Intelligence Blacklist: Automatically reports confirmed fake URLs to a central database; once a URL reaches a threshold, all users receive instant "Danger" alerts upon visiting.

Admin Dashboard: A web-based interface to monitor detection history and manage the global blacklist.

 Technical Architecture
Backend: FastAPI, PyTorch, Uvicorn, SQLite3.

Models: AutoencoderKL (Stability AI), LPIPS (VGG-based).

Frontend (Extension): Vanilla JavaScript, Chrome Extension API (Manifest V3).

Communication: Secure Background Service Worker proxying for HTTPS/HTTP compatibility.

 Getting Started
Prerequisites
Python 3.10+

CUDA-enabled GPU (recommended) or CPU

Google Chrome Browser

1. Server Setup
Clone the repository and install dependencies:

```Bash
pip install fastapi uvicorn torch torchvision pillow-heif diffusers lpips pandas opencv-python  python-multipart accelerate python-magic-bin c2pa-python piexif
```
Place your pre-trained models in the local_models/ directory.

Run the server:

```bash
python integrated_server.py
```
The server will be accessible at http://localhost:80 (or your configured domain).

2. Extension Installation
Open Chrome and navigate to chrome://extensions.

Enable "Developer mode" in the top right.

Click "Load unpacked" and select the extension folder containing manifest.json.

Update the SERVER_BASE in background.js to match your server's URL.

 Usage
Automatic Alert: If you visit a URL previously flagged by the community, an alert will appear at the top of your screen.
YouTube Analysis: Click the  Check for Deepfake button located near the video title.
Image Analysis: Hover over any image on the web and click the  floating icon.
Dashboard: Access http://your-server-ip/admin to view and manage detection records.
