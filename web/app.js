const input = document.querySelector('#video-input');
const dropzone = document.querySelector('#dropzone');

//cache the page controls once so event handlers and render functions can update the same elements
//the travel detector uses its own frame limit, separate from the sample count used by other detectors
const travelFrameCount = document.querySelector('#travel-frame-count');
const fileName = document.querySelector('#file-name');
const analyzeButton = document.querySelector('#analyze-button');
const previewSection = document.querySelector('#preview-section');
const preview = document.querySelector('#video-preview');
const resultsSection = document.querySelector('#results-section');
const resultsGrid = document.querySelector('#results-grid');
const frameCount = document.querySelector('#frame-count');
const message = document.querySelector('#message');
let selectedFile = null;
let previewUrl = null;

//update the shared status area so upload and analysis feedback appears in one place
//the optional type adds the matching success or error styling without changing the message text
function showMessage(text, type = '') {
  message.textContent = text;
  message.className = `message ${type}`;
}

//validate a chosen video, replace its preview URL, and reset results from an older upload
//keeping the selected File lets the analyze action upload the original data rather than the preview
function selectFile(file) {
  if (!file || !file.type.startsWith('video/')) {
    showMessage('Please choose a supported video file.', 'error');
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  analyzeButton.disabled = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  //use a local object URL for playback and release the previous one to avoid retaining old file data
  preview.src = previewUrl;
  previewSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  showMessage('Clip loaded and ready for review.', 'success');
}

//use the same validation and preview flow for files selected in the picker or dropped onto the page
input.addEventListener('change', () => selectFile(input.files[0]));
//prevent the browser's default file-open behavior and highlight the drop target while a file is dragged
['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add('active');
}));
//remove the highlight when the pointer leaves or completes a drop, whether or not a file was accepted
['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove('active');
}));
dropzone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

//submit the selected clip and sampling preference, then show the response or an error
//disable the button during the request so the same upload is not submitted multiple times
analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = 'Analyzing frames...';
  showMessage('Sampling footage and running available detectors...');
  const formData = new FormData();
  //send the video and selected travel-detector frame limit together as multipart form data
  formData.append('video', selectedFile);
  formData.append('travel_frames', travelFrameCount.value);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: formData });
    const payload = await response.json();
    //surface API-provided failure details when available instead of rendering an unsuccessful response
    if (!response.ok) throw new Error(payload.error || 'Analysis failed.');
    renderResults(payload);
    showMessage('Review complete.', 'success');
  } catch (error) {
    //keep the failure visible in the shared status area so the user can correct the issue and retry
    showMessage(error.message, 'error');
  } finally {
    //restore the button after either success or failure so another review can be started
    analyzeButton.disabled = false;
    analyzeButton.textContent = 'Analyze video';
  }
});

//turn the analysis response into a summary and one result card per detector
//each card can show either a detector error or its prediction, with timing details when provided
function renderResults(payload) {
  frameCount.textContent = `${payload.frame_count} sampled · decoded in ${Math.round(payload.decoding_ms)} ms · travel setting ${payload.travel_frame_limit}`;
  resultsGrid.innerHTML = payload.results.map((result) => {
    //display confidence as a whole percentage and use zero when the detector has no score
    const confidence = result.confidence == null ? 0 : Math.round(result.confidence * 100);
    const state = result.error ? `<div class="result-error">${result.error}</div>` : `<div class="metric-label">Likely result</div><div class="metric-value">${result.label}</div><div class="progress"><span style="width: ${confidence}%"></span></div><div class="confidence">${confidence}% confidence</div>`;
    //show stage timings to compare travel frame-count settings
    const timing = result.timings_ms ? `<div class="timing-summary"><span>${result.timings_ms.processed_frames} frames</span><span>prep ${Math.round(result.timings_ms.preprocessing_ms)} ms</span><span>predict ${Math.round(result.timings_ms.inference_ms)} ms</span><span>load ${Math.round(result.timings_ms.model_load_ms)} ms</span></div>` : '';
    return `<article class="result-card"><div class="card-top"><h3>${result.name}</h3><span class="card-line"></span></div><p>${result.description}</p>${state}${timing}</article>`;
  }).join('');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

//load detector availability for the sidebar so users can see which models can run
//status is separate from video analysis, allowing readiness to appear before any clip is uploaded
async function loadDetectorStatus() {
  try {
    const response = await fetch('/api/status');
    const payload = await response.json();
    document.querySelector('#detector-status').innerHTML = payload.detectors.map((detector) => `<div class="detector-status"><div><strong>${detector.name}</strong><span>${detector.description}</span></div><i class="status-dot ${detector.available ? 'ready' : 'missing'}"></i><small>${detector.available ? 'Ready' : 'Not trained / missing'}</small></div>`).join('');
  } catch (error) {
    //show a clear unavailable state in the sidebar if the status endpoint cannot be reached
    document.querySelector('#detector-status').innerHTML = '<div class="detector-status"><span>API unavailable</span></div>';
  }
}

//populate the sidebar as soon as the page is ready to use
loadDetectorStatus();
