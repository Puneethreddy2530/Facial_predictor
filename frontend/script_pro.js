const defaultApi = 'http://127.0.0.1:8001';
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

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
const resultsDiv = document.getElementById('results');
const resultContent = document.getElementById('result-content');
const processingInfo = document.getElementById('processingInfo');
const multiResults = document.getElementById('multiResults');

let mediaStream = null;
let lastCaptureUrl = null;

// Load saved API URL or use default
apiUrlInput.value = localStorage.getItem('apiUrl') || defaultApi;

// File input change handler with validation
imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  
  // Validate file size
  if (file.size > MAX_FILE_SIZE) {
    showStatus('File too large! Maximum size is 10MB.', 'error');
    imageInput.value = '';
    return;
  }
  
  // Validate file type
  if (!file.type.startsWith('image/')) {
    showStatus('Please select a valid image file (JPG, PNG, etc.)', 'error');
    imageInput.value = '';
    return;
  }
  
  const url = URL.createObjectURL(file);
  preview.src = url;
  lastCaptureUrl = url;
  resultsDiv.classList.add('hidden');
  showStatus('Image loaded. Click "Analyze Photo" to process.', 'success');
});

// Save API URL
saveApiBtn.addEventListener('click', () => {
  const url = apiUrlInput.value.trim();
  if (url) {
    localStorage.setItem('apiUrl', url);
    showStatus('API URL saved!', 'success');
  }
});

// Camera controls
startCamBtn.addEventListener('click', async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      } 
    });
    videoEl.srcObject = mediaStream;
    videoEl.classList.remove('hidden');
    stopCamBtn.classList.remove('hidden');
    captureBtn.classList.remove('hidden');
    startCamBtn.classList.add('hidden');
    showStatus('Camera started. Position yourself and click "Capture Photo".', 'success');
  } catch (err) {
    console.error('Camera error:', err);
    showStatus('Camera access denied or unavailable: ' + err.message, 'error');
  }
});

stopCamBtn.addEventListener('click', () => {
  stopCamera();
  showStatus('Camera stopped.', 'info');
});

captureBtn.addEventListener('click', () => {
  if (!mediaStream) return;
  
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0);
  
  canvas.toBlob((blob) => {
    const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
    setFileFromBlob(file);
    showStatus('Photo captured! Click "Analyze Photo" to process.', 'success');
  }, 'image/jpeg', 0.95);
  
  stopCamera();
});

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  videoEl.classList.add('hidden');
  videoEl.srcObject = null;
  stopCamBtn.classList.add('hidden');
  captureBtn.classList.add('hidden');
  startCamBtn.classList.remove('hidden');
}

function setFileFromBlob(file) {
  const dt = new DataTransfer();
  dt.items.add(file);
  imageInput.files = dt.files;
  
  const url = URL.createObjectURL(file);
  preview.src = url;
  lastCaptureUrl = url;
  resultsDiv.classList.add('hidden');
}

// Form submission
uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const file = imageInput.files[0];
  if (!file) {
    showStatus('Please select or capture an image first.', 'error');
    return;
  }
  
  const apiUrl = apiUrlInput.value.trim() || defaultApi;
  
  loader.classList.remove('hidden');
  resultsDiv.classList.add('hidden');
  showStatus('Analyzing image...', 'info');
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const startTime = Date.now();
    const response = await fetch(`${apiUrl}/predict`, {
      method: 'POST',
      body: formData
    });
    
    const clientTime = Date.now() - startTime;
    
    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `Server error: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error('Analysis failed. Please try again.');
    }
    
    displayResults(data, clientTime);
    showStatus(`✓ Analysis complete! Found ${data.faces_detected} face(s).`, 'success');
    
  } catch (err) {
    console.error('Analysis error:', err);
    showStatus('Error: ' + err.message, 'error');
    resultsDiv.classList.add('hidden');
  } finally {
    loader.classList.add('hidden');
  }
});

function displayResults(data, clientTime) {
  // Processing info
  processingInfo.innerHTML = `
    <div class="processing-stats">
      <div class="stat">
        <span class="stat-label">Server Time:</span>
        <span class="stat-value">${data.processing_time_ms.toFixed(0)}ms</span>
      </div>
      <div class="stat">
        <span class="stat-label">Total Time:</span>
        <span class="stat-value">${clientTime}ms</span>
      </div>
      <div class="stat">
        <span class="stat-label">Faces Detected:</span>
        <span class="stat-value">${data.faces_detected}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Image Size:</span>
        <span class="stat-value">${data.metadata.image_dimensions.width}×${data.metadata.image_dimensions.height}</span>
      </div>
    </div>
  `;
  
  // Warnings
  if (data.warnings && data.warnings.length > 0) {
    const warningsHtml = data.warnings.map(w => 
      `<div class="warning-item">⚠️ ${w}</div>`
    ).join('');
    processingInfo.innerHTML += `<div class="warnings">${warningsHtml}</div>`;
  }
  
  // Display each face
  if (data.faces && data.faces.length > 0) {
    multiResults.innerHTML = data.faces.map(face => createFaceCard(face)).join('');
  } else {
    multiResults.innerHTML = '<div class="no-faces">No faces detected in the image.</div>';
  }
  
  resultsDiv.classList.remove('hidden');
  resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createFaceCard(face) {
  const qualityClass = face.quality.quality_score >= 70 ? 'good' : 
                       face.quality.quality_score >= 40 ? 'medium' : 'poor';
  
  return `
    <div class="face-card">
      <div class="face-header">
        <h3>Face ${face.face_id + 1}</h3>
        <div class="quality-badge ${qualityClass}">
          Quality: ${face.quality.quality_score.toFixed(0)}%
        </div>
      </div>
      
      <div class="face-details">
        <div class="detail-section">
          <h4>📊 Demographics</h4>
          <div class="detail-item">
            <span class="label">Age:</span>
            <span class="value"><strong>${face.age}</strong> years (${face.age_range})</span>
          </div>
          <div class="detail-item">
            <span class="label">Gender:</span>
            <span class="value"><strong>${face.gender}</strong> 
              <span class="confidence">(${(face.gender_confidence * 100).toFixed(1)}%)</span>
            </span>
          </div>
          <div class="detail-item">
            <span class="label">Ethnicity:</span>
            <span class="value"><strong>${face.dominant_race}</strong></span>
          </div>
        </div>
        
        <div class="detail-section">
          <h4>😊 Emotions</h4>
          <div class="emotion-bars">
            ${createEmotionBars(face.emotion_scores, face.dominant_emotion)}
          </div>
        </div>
        
        <div class="detail-section">
          <h4>🌍 Ethnicity Distribution</h4>
          <div class="race-bars">
            ${createRaceBars(face.race_scores, face.dominant_race)}
          </div>
        </div>
        
        <div class="detail-section">
          <h4>🔍 Image Quality</h4>
          <div class="quality-metrics">
            <div class="metric">
              <span class="metric-label">Sharpness:</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${face.quality.sharpness}%"></div>
              </div>
              <span class="metric-value">${face.quality.sharpness.toFixed(0)}%</span>
            </div>
            <div class="metric">
              <span class="metric-label">Brightness:</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${face.quality.brightness}%"></div>
              </div>
              <span class="metric-value">${face.quality.brightness.toFixed(0)}%</span>
            </div>
            <div class="metric">
              <span class="metric-label">Contrast:</span>
              <div class="metric-bar">
                <div class="metric-fill" style="width: ${face.quality.contrast}%"></div>
              </div>
              <span class="metric-value">${face.quality.contrast.toFixed(0)}%</span>
            </div>
          </div>
        </div>
        
        <div class="detail-section">
          <h4>📍 Face Location</h4>
          <div class="detail-item">
            <span class="label">Position:</span>
            <span class="value">x=${face.face_location.x}, y=${face.face_location.y}</span>
          </div>
          <div class="detail-item">
            <span class="label">Size:</span>
            <span class="value">${face.face_location.w} × ${face.face_location.h}px</span>
          </div>
          <div class="detail-item">
            <span class="label">Confidence:</span>
            <span class="value"><strong>${(face.face_location.confidence * 100).toFixed(1)}%</strong></span>
          </div>
        </div>
      </div>
    </div>
  `;
}

function createEmotionBars(emotions, dominant) {
  const emotionIcons = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😮',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
  };
  
  return Object.entries(emotions)
    .sort((a, b) => b[1] - a[1])
    .map(([emotion, score]) => {
      const isDominant = emotion === dominant;
      return `
        <div class="bar-item ${isDominant ? 'dominant' : ''}">
          <span class="bar-label">${emotionIcons[emotion]} ${capitalizeFirst(emotion)}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${score}%"></div>
          </div>
          <span class="bar-value">${score.toFixed(1)}%</span>
        </div>
      `;
    }).join('');
}

function createRaceBars(races, dominant) {
  return Object.entries(races)
    .sort((a, b) => b[1] - a[1])
    .map(([race, score]) => {
      const isDominant = race === dominant.toLowerCase();
      return `
        <div class="bar-item ${isDominant ? 'dominant' : ''}">
          <span class="bar-label">${capitalizeFirst(race)}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${score}%"></div>
          </div>
          <span class="bar-value">${score.toFixed(1)}%</span>
        </div>
      `;
    }).join('');
}

function capitalizeFirst(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function showStatus(message, type = 'info') {
  statusEl.textContent = message;
  statusEl.className = 'status ' + type;
  statusEl.classList.remove('hidden');
  
  if (type === 'success' || type === 'error') {
    setTimeout(() => {
      statusEl.classList.add('hidden');
    }, 5000);
  }
}

// Initialize
showStatus('Ready. Upload or capture an image to begin.', 'info');
