const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const uploadForm = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const loader = document.getElementById('loader');
const apiUrlInput = document.getElementById('apiUrl');
const saveApiBtn = document.getElementById('saveApi');
const startCamBtn = document.getElementById('startCamera');
const stopCamBtn = document.getElementById('stopCamera');
const captureBtn = document.getElementById('capturePhoto');
const videoEl = document.getElementById('cameraStream');
const ageVal = document.getElementById('ageVal');
const ageBand = document.getElementById('ageBand');
const genderVal = document.getElementById('genderVal');
const emotionVal = document.getElementById('emotionVal');
const raceVal = document.getElementById('raceVal');
const emotionJson = document.getElementById('emotionJson');
const raceJson = document.getElementById('raceJson');

let mediaStream = null;
let lastCaptureUrl = null;

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  lastCaptureUrl = url;
});

// Load API URL from query (?api=) or localStorage
const urlParam = new URLSearchParams(location.search).get('api');
const savedApi = localStorage.getItem('apiUrl');
const defaultApi = 'http://127.0.0.1:8001';
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
apiUrlInput && (apiUrlInput.value = urlParam || savedApi || defaultApi);
saveApiBtn && saveApiBtn.addEventListener('click', () => {
  localStorage.setItem('apiUrl', apiUrlInput.value.trim());
  statusEl && (statusEl.textContent = ' API URL saved');
  statusEl.style.color = '#4ade80';
  setTimeout(() => { statusEl.textContent = ''; }, 2000);
});

async function startCamera() {
  try {
    if (mediaStream) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    videoEl.srcObject = mediaStream;
    videoEl.classList.remove('hidden');
    await videoEl.play();
    statusEl && (statusEl.textContent = 'Camera is live');
    statusEl.style.color = '#4ade80';
  } catch (err) {
    console.error('Camera error:', err);
    statusEl && (statusEl.textContent = '❌ Cannot access camera: ' + err.message);
    statusEl.style.color = '#ef4444';
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  videoEl && videoEl.classList.add('hidden');
}

function setFileFromBlob(blob) {
  const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
  const dt = new DataTransfer();
  dt.items.add(file);
  imageInput.files = dt.files;
}

async function captureFrame() {
  if (!mediaStream || !videoEl || videoEl.readyState < 2) {
    statusEl && (statusEl.textContent = 'Start the camera before capturing');
    statusEl.style.color = '#f59e0b';
    return;
  }
  const w = videoEl.videoWidth || 640;
  const h = videoEl.videoHeight || 480;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(videoEl, 0, 0, w, h);
  canvas.toBlob((blob) => {
    if (!blob) return;
    setFileFromBlob(blob);
    const url = URL.createObjectURL(blob);
    preview.src = url;
    lastCaptureUrl = url;
    statusEl && (statusEl.textContent = 'Captured from camera');
    statusEl.style.color = '#4ade80';
  }, 'image/jpeg', 0.92);
}

startCamBtn && startCamBtn.addEventListener('click', startCamera);
stopCamBtn && stopCamBtn.addEventListener('click', () => {
  stopCamera();
  statusEl && (statusEl.textContent = 'Camera stopped');
  statusEl.style.color = '#a9b3c1';
});
captureBtn && captureBtn.addEventListener('click', captureFrame);

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    alert('Please choose an image first');
    return;
  }

  const base = (apiUrlInput ? apiUrlInput.value : (savedApi || defaultApi)).trim() || defaultApi;
  
  if (!base) {
    statusEl && (statusEl.textContent = '⚠️ Backend API URL not configured.');
    statusEl.style.color = '#ef4444';
    return;
  }

  const fd = new FormData();
  fd.append('file', file);

  statusEl && (statusEl.textContent = '');
  loader && loader.classList.remove('hidden');
  const loaderText = document.querySelector('#loader div:last-child');
  let startTs = Date.now();
  const updateLoader = (msg) => { if (loaderText) loaderText.textContent = msg; };
  updateLoader('Warming up backend (free tier) …');

  try {
    const apiUrl = base.replace(/\/$/, '') + '/predict';
    const controller = new AbortController();
    const timeoutMs = 90000; // 90s to survive cold starts
    const timer = setTimeout(() => controller.abort('Request timed out after 90s'), timeoutMs);

    const doRequest = async () => {
      const res = await fetch(apiUrl, { method: 'POST', body: fd, signal: controller.signal });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status} ${res.statusText}: ${txt.slice(0,200)}`);
      }
      return res.json();
    };

    let data;
    try {
      updateLoader('Analyzing image …');
      data = await doRequest();
    } catch (err) {
      // One auto-retry after small backoff (handles wake-ups)
      updateLoader('Backend waking up… retrying once …');
      await new Promise(r => setTimeout(r, 2500));
      data = await doRequest();
    } finally {
      clearTimeout(timer);
    }

    if (data.error) {
      statusEl && (statusEl.textContent = '⚠️ API Error: ' + data.error);
      statusEl.style.color = '#f59e0b';
      console.error('API Error:', data.error);
    }

    renderResults(data);
  } catch (err) {
    console.error('Fetch error:', err);
    const elapsed = ((Date.now() - startTs) / 1000).toFixed(1);
    if (err.name === 'AbortError' || err.message.includes('timed out')) {
      statusEl && (statusEl.textContent = `❌ Request timed out after ${elapsed}s. The free tier may be waking up; please try again.`);
    } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      statusEl && (statusEl.textContent = '❌ Cannot connect to backend. The free tier may be sleeping (takes 30-60s to wake up). Please wait and try again.');
    } else {
      statusEl && (statusEl.textContent = '❌ Error: ' + err.message);
    }
    statusEl.style.color = '#ef4444';
  } finally {
    loader && loader.classList.add('hidden');
    updateLoader('Analyzing with DeepFace models…');
  }
});

function renderResults(data) {
  const age = Math.round(data.age);
  let band = 'Adult';
  if (age < 2) band = 'Infant';
  else if (age < 13) band = 'Child';
  else if (age < 18) band = 'Teen';
  else if (age > 60) band = 'Senior';

  ageVal && (ageVal.textContent = age);
  ageBand && (ageBand.textContent = band);
  genderVal && (genderVal.textContent = data.gender);
  emotionVal && (emotionVal.textContent = data.dominant_emotion);
  raceVal && (raceVal.textContent = data.dominant_race);

  emotionJson && (emotionJson.textContent = JSON.stringify(data.emotion, null, 2));
  raceJson && (raceJson.textContent = JSON.stringify(data.race, null, 2));

  // Quality and confidence warnings
  let warnings = [];
  if (data.low_quality && data.quality_reason) {
    warnings.push(' ' + data.quality_reason);
  }
  if (data.low_confidence) {
    warnings.push('ℹ Low confidence in emotion/race — results may be uncertain.');
  }
  if (statusEl) {
    statusEl.textContent = warnings.join(' ');
    statusEl.style.color = warnings.length > 0 ? '#ffcc00' : '#4ade80';
  }
}
