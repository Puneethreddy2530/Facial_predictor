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
const defaultApi = 'http://localhost:8001';
apiUrlInput && (apiUrlInput.value = urlParam || savedApi || defaultApi);
saveApiBtn && saveApiBtn.addEventListener('click', () => {
  localStorage.setItem('apiUrl', apiUrlInput.value.trim());
  statusEl && (statusEl.textContent = 'API saved');
});

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    alert('Please choose an image first');
    return;
  }

  const fd = new FormData();
  fd.append('file', file);

  statusEl && (statusEl.textContent = '');
  loader && loader.classList.remove('hidden');

  try {
    const base = (apiUrlInput ? apiUrlInput.value : (savedApi || defaultApi));
    const apiUrl = base.replace(/\/$/, '') + '/predict';
    
    const res = await fetch(apiUrl, {
      method: 'POST',
      body: fd,
    });

    if (!res.ok) {
      throw new Error(HTTP : );
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    console.error('Fetch error:', err);
    statusEl && (statusEl.textContent = 'Error: ' + err.message + ' — verify API URL and backend is running.');
  }
  loader && loader.classList.add('hidden');
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
    statusEl.style.color = warnings.length > 0 ? '#ffcc00' : '';
  }
}
