const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const uploadForm = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const loader = document.getElementById('loader');
const apiUrlInput = document.getElementById('apiUrl');
const saveApiBtn = document.getElementById('saveApi');
const ageVal = document.getElementById('ageVal');
const ageBand = document.getElementById('ageBand');
const genderVal = document.getElementById('genderVal');
const emotionVal = document.getElementById('emotionVal');
const raceVal = document.getElementById('raceVal');
const emotionJson = document.getElementById('emotionJson');
const raceJson = document.getElementById('raceJson');

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
});

// Load API URL from query (?api=) or localStorage
const urlParam = new URLSearchParams(location.search).get('api');
const savedApi = localStorage.getItem('apiUrl');
const defaultApi = 'https://facial-predictor-api.onrender.com';
apiUrlInput && (apiUrlInput.value = urlParam || savedApi || defaultApi);
saveApiBtn && saveApiBtn.addEventListener('click', () => {
  localStorage.setItem('apiUrl', apiUrlInput.value.trim());
  statusEl && (statusEl.textContent = ' API URL saved');
  statusEl.style.color = '#4ade80';
  setTimeout(() => { statusEl.textContent = ''; }, 2000);
});

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
