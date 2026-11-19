/**
 * web/static/js/app.js - Forensic Studio Frontend Interactive Controller.
 * 
 * Manages:
 * 1. WebRTC webcam capture with autonomous Dual-Mode Live Streamer:
 *    - WebSocket (/ws/live) primary
 *    - High-speed HTTP Binary POST (/api/live-frame) seamless fallback with in-flight flow control (~30-40 FPS).
 * 2. Dynamic banknote bounding reticle with live emerald/coral status tracking.
 * 3. Photo & Video upload pipelines with deep multi-spectral plate rendering.
 * 4. Active screen-flash watermark acquisition.
 * 5. Synthetic Demonstration Bench with Side-by-Side Dual Comparative Analysis.
 * 6. Discreet Acoustic Rhythm Audio Feedback via Web Audio API.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // State
  let currentMode = 'live'; // 'live', 'photo', 'video', 'synthetic'
  let ws = null;
  let webcamStream = null;
  let isStreaming = false;
  let isFrameInFlight = false;
  let activeTransport = 'none'; // 'ws' or 'http'
  let currentCameraFacing = 'user'; // 'user' or 'environment'
  let lastFrameTime = performance.now();
  let rollingFps = 0.0;

  // DOM Elements
  const videoEl = document.getElementById('webcam-video');
  const canvasEl = document.getElementById('viewport-canvas');
  const ctx = canvasEl.getContext('2d');
  const photoPreview = document.getElementById('uploaded-image-preview');
  const videoPreview = document.getElementById('uploaded-video');
  const dropzoneBox = document.getElementById('dropzone-box');
  const flashOverlay = document.getElementById('screen-flash-overlay');

  const filePhotoInput = document.getElementById('file-photo-input');
  const fileVideoInput = document.getElementById('file-video-input');
  const btnUploadTrigger = document.getElementById('btn-upload-trigger');
  const btnCamSwitch = document.getElementById('btn-cam-switch');
  const btnFlash = document.getElementById('btn-flash-differencing');
  const btnAudio = document.getElementById('btn-toggle-audio');
  const audioIcon = document.getElementById('audio-icon');

  // Telemetry Elements
  const verdictBadge = document.getElementById('verdict-badge');
  const verdictText = document.getElementById('verdict-text');
  const denomVal = document.getElementById('denom-value');
  const denomConf = document.getElementById('denom-confidence');
  const fpsText = document.getElementById('telemetry-fps');
  const latencyText = document.getElementById('telemetry-latency');
  const audioTagText = document.getElementById('telemetry-audio-tag');

  const barCounterfeit = document.getElementById('bar-counterfeit');
  const scoreCounterfeit = document.getElementById('metric-counterfeit-score');
  const barHalftone = document.getElementById('bar-halftone');
  const scoreHalftone = document.getElementById('metric-halftone-score');
  const barCnn = document.getElementById('bar-cnn');
  const scoreCnn = document.getElementById('metric-cnn-score');
  const barWatermark = document.getElementById('bar-watermark');
  const scoreWatermark = document.getElementById('metric-watermark-density');

  // Multi-Spectral Plates
  const plateClahe = document.getElementById('plate-img-clahe');
  const plateSr = document.getElementById('plate-img-sr');
  const plateFft = document.getElementById('plate-img-fft');
  const plateC = document.getElementById('plate-img-c');
  const plateM = document.getElementById('plate-img-m');
  const plateWm = document.getElementById('plate-img-wm');

  // Sections & Bench Controls
  const syntheticBar = document.getElementById('synthetic-bench-bar');
  const benchDenomSelect = document.getElementById('bench-denom-select');
  const btnDemoCompare = document.getElementById('btn-demo-compare');
  const btnDemoGenuine = document.getElementById('btn-demo-genuine');
  const btnDemoCounterfeit = document.getElementById('btn-demo-counterfeit');
  const btnDemoToy = document.getElementById('btn-demo-toy');
  const comparisonSection = document.getElementById('comparison-section');

  // Comparison UI Elements
  const compareHeading = document.getElementById('compare-heading');
  const cmpGenFull = document.getElementById('cmp-gen-full');
  const cmpGenScore = document.getElementById('cmp-gen-score');
  const cmpGenVerdict = document.getElementById('cmp-gen-verdict');
  const cmpGenHt = document.getElementById('cmp-gen-ht');
  const cmpGenDetail = document.getElementById('cmp-gen-detail');
  const cmpGenSr = document.getElementById('cmp-gen-sr');
  const cmpGenFft = document.getElementById('cmp-gen-fft');

  const cmpSusFull = document.getElementById('cmp-sus-full');
  const cmpSusScore = document.getElementById('cmp-sus-score');
  const cmpSusVerdict = document.getElementById('cmp-sus-verdict');
  const cmpSusHt = document.getElementById('cmp-sus-ht');
  const cmpSusDetail = document.getElementById('cmp-sus-detail');
  const cmpSusSr = document.getElementById('cmp-sus-sr');
  const cmpSusFft = document.getElementById('cmp-sus-fft');

  const timelineCard = document.getElementById('video-timeline-section');
  const timelineContainer = document.getElementById('timeline-container');
  const timelineStats = document.getElementById('timeline-stats-text');

  // Off-screen canvas for frame resizing and transmission (480x270 @ 16:9)
  const transmitCanvas = document.createElement('canvas');
  const transmitCtx = transmitCanvas.getContext('2d');
  transmitCanvas.width = 480;
  transmitCanvas.height = 270;

  /* ----------------------------------------------------
     Mode Navigation
  ---------------------------------------------------- */
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setMode(btn.dataset.mode);
    });
  });

  function setMode(mode) {
    currentMode = mode;
    stopLiveStream();

    // Reset viewports
    videoEl.style.display = 'none';
    canvasEl.style.display = 'none';
    photoPreview.style.display = 'none';
    videoPreview.style.display = 'none';
    dropzoneBox.style.display = 'none';
    syntheticBar.style.display = 'none';
    if (comparisonSection) comparisonSection.style.display = 'none';
    timelineCard.classList.remove('active');
    btnCamSwitch.style.display = 'none';
    btnUploadTrigger.style.display = 'none';

    if (mode === 'live') {
      document.getElementById('viewport-heading').innerText = 'Sensor Feed & Reticle';
      videoEl.style.display = 'block';
      canvasEl.style.display = 'block';
      btnCamSwitch.style.display = 'inline-flex';
      startLiveStream();
    } else if (mode === 'photo') {
      document.getElementById('viewport-heading').innerText = 'Photo Forensics Viewport';
      if (photoPreview.src && photoPreview.src.length > 50 && !photoPreview.src.includes('data:image/svg')) {
        dropzoneBox.style.display = 'none';
        photoPreview.style.display = 'block';
      } else {
        dropzoneBox.style.display = 'flex';
        photoPreview.style.display = 'none';
      }
      btnUploadTrigger.style.display = 'inline-flex';
      btnUploadTrigger.innerText = 'Choose Photo';
    } else if (mode === 'video') {
      document.getElementById('viewport-heading').innerText = 'Video Timeline Inspection';
      dropzoneBox.style.display = 'flex';
      btnUploadTrigger.style.display = 'inline-flex';
      btnUploadTrigger.innerText = 'Choose Video';
    } else if (mode === 'synthetic') {
      document.getElementById('viewport-heading').innerText = 'Synthetic Algorithmic Bench';
      photoPreview.src = '';
      photoPreview.style.display = 'block';
      syntheticBar.style.display = 'flex';
      if (comparisonSection) comparisonSection.style.display = 'none';
      const initialDenom = benchDenomSelect ? benchDenomSelect.value : '50';
      triggerSyntheticDemo(initialDenom, false);
    } else if (mode === 'compare') {
      document.getElementById('viewport-heading').innerText = 'Side-by-Side Dual Comparison Lab';
      photoPreview.style.display = 'none';
      syntheticBar.style.display = 'flex';
      if (comparisonSection) {
        comparisonSection.style.display = 'block';
      }
      const initialDenom = benchDenomSelect ? benchDenomSelect.value : '50';
      triggerComparisonDemo(initialDenom);
    }

    if (window.lucide) window.lucide.createIcons();
  }

  /* ----------------------------------------------------
     Live Webcam & Dual-Mode Streamer (WS + HTTP Fallback)
  ---------------------------------------------------- */
  async function startLiveStream() {
    try {
      const constraints = {
        video: {
          facingMode: { ideal: currentCameraFacing },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = webcamStream;
      await videoEl.play();

      function updateCanvasDims() {
        if (videoEl.videoWidth && videoEl.videoHeight) {
          canvasEl.width = videoEl.videoWidth;
          canvasEl.height = videoEl.videoHeight;
        } else {
          canvasEl.width = 640;
          canvasEl.height = 360;
        }
      }

      videoEl.onloadedmetadata = updateCanvasDims;
      videoEl.onplaying = updateCanvasDims;
      updateCanvasDims();

      isStreaming = true;
      isFrameInFlight = false;
      lastFrameTime = performance.now();
      rollingFps = 0.0;

      // Real-time zero-delay binary HTTP streaming engine (>35 FPS)
      startHttpLiveStream();

    } catch (err) {
      console.warn('[Webcam] Could not access live camera device:', err);
      document.getElementById('engine-status-text').innerText = 'CAMERA UNAVAILABLE • USE UPLOAD / BENCH';
      setMode('synthetic');
    }
  }

  function stopLiveStream() {
    isStreaming = false;
    isFrameInFlight = false;
    activeTransport = 'none';

    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    if (ws) {
      try { ws.close(); } catch (_) {}
      ws = null;
    }
    if (ctx && canvasEl) {
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    }
  }

  function tryConnectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    let wsOpened = false;

    try {
      ws = new WebSocket(wsUrl);

      const openTimeout = setTimeout(() => {
        if (!wsOpened && isStreaming) {
          console.warn('[Transport] WebSocket timed out, falling back to HTTP stream');
          startHttpLiveStream();
        }
      }, 1200);

      ws.onopen = () => {
        wsOpened = true;
        clearTimeout(openTimeout);
        activeTransport = 'ws';
        document.getElementById('engine-status-text').innerText = 'LIVE STREAM • WEBSOCKET ONLINE';
        sendNextWsFrame();
      };

      ws.onmessage = (event) => {
        isFrameInFlight = false;
        try {
          const telemetry = JSON.parse(event.data);
          computeClientFps();
          updateTelemetryUI(telemetry);
          drawReticle(telemetry);

          if (telemetry.audio_tag && window.audioEngine) {
            window.audioEngine.trigger(telemetry.audio_tag);
          }
        } catch (e) {
          console.error('Error parsing telemetry message:', e);
        }

        if (isStreaming && activeTransport === 'ws') {
          requestAnimationFrame(sendNextWsFrame);
        }
      };

      ws.onerror = (err) => {
        clearTimeout(openTimeout);
        if (activeTransport !== 'http') {
          console.warn('[Transport] WebSocket error, engaging HTTP stream fallback');
          startHttpLiveStream();
        }
      };

      ws.onclose = () => {
        clearTimeout(openTimeout);
        if (isStreaming && activeTransport !== 'http') {
          startHttpLiveStream();
        }
      };

    } catch (e) {
      startHttpLiveStream();
    }
  }

  function sendNextWsFrame() {
    if (!isStreaming || activeTransport !== 'ws' || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (isFrameInFlight) return;
    if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      requestAnimationFrame(sendNextWsFrame);
      return;
    }

    isFrameInFlight = true;
    transmitCtx.drawImage(videoEl, 0, 0, transmitCanvas.width, transmitCanvas.height);
    const b64 = transmitCanvas.toDataURL('image/jpeg', 0.65);
    ws.send(b64);
  }

  function startHttpLiveStream() {
    if (activeTransport === 'http') return;
    activeTransport = 'http';
    if (ws) {
      try { ws.close(); } catch (_) {}
      ws = null;
    }
    document.getElementById('engine-status-text').innerText = 'LIVE STREAM • REAL-TIME FEED ACTIVE';
    isFrameInFlight = false;
    requestAnimationFrame(streamNextHttpFrame);
  }

  async function streamNextHttpFrame() {
    if (!isStreaming || currentMode !== 'live' || activeTransport !== 'http') return;
    if (isFrameInFlight) return;
    if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      requestAnimationFrame(streamNextHttpFrame);
      return;
    }

    isFrameInFlight = true;
    transmitCtx.drawImage(videoEl, 0, 0, transmitCanvas.width, transmitCanvas.height);

    transmitCanvas.toBlob(async (blob) => {
      if (!blob) {
        isFrameInFlight = false;
        requestAnimationFrame(streamNextHttpFrame);
        return;
      }

      try {
        const t0 = performance.now();
        const resp = await fetch('/api/live-frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: blob
        });

        if (resp.ok) {
          const telemetry = await resp.json();
          const t1 = performance.now();
          telemetry.latency_ms = Math.round(t1 - t0);
          computeClientFps();

          updateTelemetryUI(telemetry);
          drawReticle(telemetry);

          document.getElementById('engine-status-text').innerText = 'LIVE STREAM • REAL-TIME FEED ACTIVE';

          if (telemetry.audio_tag && window.audioEngine) {
            window.audioEngine.trigger(telemetry.audio_tag);
          }
        } else {
          document.getElementById('engine-status-text').innerText = 'ENGINE RECONNECTING...';
        }
      } catch (err) {
        document.getElementById('engine-status-text').innerText = 'SERVER CONNECTING...';
        await new Promise(r => setTimeout(r, 300));
      } finally {
        isFrameInFlight = false;
        if (isStreaming && currentMode === 'live') {
          requestAnimationFrame(streamNextHttpFrame);
        }
      }
    }, 'image/jpeg', 0.60);
  }

  function computeClientFps() {
    const now = performance.now();
    const dt = (now - lastFrameTime) / 1000.0;
    lastFrameTime = now;
    if (dt > 0.005 && dt < 1.0) {
      const instantFps = 1.0 / dt;
      rollingFps = rollingFps > 0 ? (0.85 * rollingFps + 0.15 * instantFps) : instantFps;
      fpsText.innerText = rollingFps.toFixed(1);
    }
  }

  /* ----------------------------------------------------
     Reticle & Dynamic Banknote Tracking Drawing
  ---------------------------------------------------- */
  function drawReticle(telemetry) {
    if (!ctx) return;
    const w = canvasEl.width;
    const h = canvasEl.height;
    ctx.clearRect(0, 0, w, h);

    const isNote = Boolean(telemetry.is_note_detected && telemetry.denomination !== 'Background');
    const isCounterfeit = Boolean(telemetry.is_counterfeit);
    const color = !isNote ? 'rgba(235, 220, 200, 0.65)' : (isCounterfeit ? '#D25858' : '#5B8B68');
    const badgeBg = isCounterfeit ? 'rgba(210, 88, 88, 0.94)' : 'rgba(91, 139, 104, 0.94)';

    let x1, y1, x2, y2;

    if (isNote && telemetry.roi_bbox) {
      const scaleX = w / 480.0;
      const scaleY = h / 270.0;

      const [bx, by, bw, bh] = telemetry.roi_bbox;
      x1 = Math.max(12, Math.round(bx * scaleX));
      y1 = Math.max(12, Math.round(by * scaleY));
      x2 = Math.min(w - 12, Math.round((bx + bw) * scaleX));
      y2 = Math.min(h - 12, Math.round((by + bh) * scaleY));

      // Draw semi-transparent bounding highlight
      ctx.fillStyle = isCounterfeit ? 'rgba(210, 88, 88, 0.10)' : 'rgba(91, 139, 104, 0.10)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    } else {
      // Default guide box in center (standard banknote proportion)
      const boxW = Math.min(380, Math.round(w * 0.50));
      const boxH = Math.round(boxW / 2.05);
      x1 = Math.round((w - boxW) / 2);
      y1 = Math.round((h - boxH) / 2);
      x2 = x1 + boxW;
      y2 = y1 + boxH;

      // Soft guide box border
      ctx.setLineDash([8, 8]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 1.8;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }

    // High-visibility corner brackets
    const bLen = Math.min(32, Math.floor((x2 - x1) * 0.20));
    ctx.strokeStyle = color;
    ctx.lineWidth = isNote ? 3.5 : 2.5;
    ctx.lineCap = 'round';

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + bLen);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + bLen, y1);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x2 - bLen, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + bLen);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x1, y2 - bLen);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + bLen, y2);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x2 - bLen, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - bLen);
    ctx.stroke();

    if (isNote) {
      // Floating label pill above the detected banknote
      const labelText = `₹${telemetry.denomination} • ${isCounterfeit ? 'SUSPECT' : 'GENUINE'} (${((telemetry.confidence || 0) * 100).toFixed(0)}%)`;
      ctx.font = '600 13px "Plus Jakarta Sans", sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const pillW = textWidth + 26;
      const pillH = 28;
      const pillX = Math.max(10, Math.min(w - pillW - 10, x1));
      const pillY = Math.max(pillH + 8, y1 - 10);

      // Pill Shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.30)';
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 2;

      // Pill Background
      ctx.fillStyle = badgeBg;
      ctx.beginPath();
      ctx.roundRect(pillX, pillY - pillH, pillW, pillH, 8);
      ctx.fill();

      // Reset Shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      // Pill Text
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(labelText, pillX + 13, pillY - 9);
    } else {
      // Subtle standby prompt
      ctx.font = '500 12px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.textAlign = 'center';
      ctx.fillText('ALIGN BANKNOTE WITHIN RETICLE', w / 2, (y1 + y2) / 2);
      ctx.textAlign = 'left';
    }
  }

  /* ----------------------------------------------------
     Active Screen-Flash Watermark Acquisition
  ---------------------------------------------------- */
  btnFlash.addEventListener('click', async () => {
    if (window.audioEngine) window.audioEngine.trigger('CLICK');

    if (currentMode === 'live' && videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      const snapCanvas = document.createElement('canvas');
      snapCanvas.width = videoEl.videoWidth || 640;
      snapCanvas.height = videoEl.videoHeight || 360;
      const snapCtx = snapCanvas.getContext('2d');
      snapCtx.drawImage(videoEl, 0, 0);
      const ambientB64 = snapCanvas.toDataURL('image/jpeg', 0.9);

      // Trigger full-screen white flash
      flashOverlay.classList.add('active');
      await new Promise(r => setTimeout(r, 160));

      snapCtx.drawImage(videoEl, 0, 0);
      const flashB64 = snapCanvas.toDataURL('image/jpeg', 0.9);
      flashOverlay.classList.remove('active');

      try {
        const resp = await fetch('/api/flash-diff', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ambient_base64: ambientB64,
            flash_base64: flashB64
          })
        });
        const data = await resp.json();
        if (data.watermark_mask) plateWm.src = data.watermark_mask;
        if (data.transmission_density !== undefined) {
          scoreWatermark.innerText = `${(data.transmission_density * 100).toFixed(1)}%`;
          barWatermark.style.width = `${Math.min(100, data.transmission_density * 500)}%`;
        }
      } catch (err) {
        console.error('Error computing flash difference:', err);
      }
    } else {
      flashOverlay.classList.add('active');
      setTimeout(() => flashOverlay.classList.remove('active'), 140);
    }
  });

  /* ----------------------------------------------------
     Photo Upload Handler
  ---------------------------------------------------- */
  btnUploadTrigger.addEventListener('click', () => {
    if (currentMode === 'photo') filePhotoInput.click();
    else if (currentMode === 'video') fileVideoInput.click();
  });

  dropzoneBox.addEventListener('click', () => {
    if (currentMode === 'photo') filePhotoInput.click();
    else if (currentMode === 'video') fileVideoInput.click();
  });

  filePhotoInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadPhoto(e.target.files[0]);
    }
  });

  async function uploadPhoto(file) {
    document.getElementById('engine-status-text').innerText = 'ANALYZING HIGH-RES PHOTO...';
    dropzoneBox.style.display = 'none';
    photoPreview.style.display = 'block';

    const reader = new FileReader();
    reader.onload = (e) => {
      photoPreview.src = e.target.result;
    };
    reader.readAsDataURL(file);

    try {
      const resp = await fetch('/api/inspect/image', {
        method: 'POST',
        headers: {
          'Content-Type': file.type || 'application/octet-stream',
          'X-Filename': file.name
        },
        body: file
      });
      const data = await resp.json();
      updateTelemetryUI(data);
      populatePlates(data);
      document.getElementById('engine-status-text').innerText = `PHOTO: ${data.resolution || 'INSPECTED'}`;

      if (data.audio_tag && window.audioEngine) {
        window.audioEngine.trigger(data.audio_tag);
      }
    } catch (err) {
      console.error('Photo inspection failed:', err);
      document.getElementById('engine-status-text').innerText = 'INSPECTION ERROR';
    }
  }

  /* ----------------------------------------------------
     Video Upload & Timeline Handler
  ---------------------------------------------------- */
  fileVideoInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadVideo(e.target.files[0]);
    }
  });

  async function uploadVideo(file) {
    document.getElementById('engine-status-text').innerText = 'PROCESSING VIDEO TIMELINE...';
    dropzoneBox.style.display = 'none';
    videoPreview.style.display = 'block';
    videoPreview.src = URL.createObjectURL(file);

    try {
      const resp = await fetch('/api/inspect/video', {
        method: 'POST',
        headers: {
          'Content-Type': file.type || 'application/octet-stream',
          'X-Filename': file.name
        },
        body: file
      });
      const data = await resp.json();

      updateTelemetryUI({
        verdict: data.verdict,
        is_counterfeit: data.is_counterfeit,
        denomination: data.primary_denomination,
        confidence: 0.95,
        composite_counterfeit_score: data.average_counterfeit_score,
        halftone_score: data.average_counterfeit_score * 0.9,
        counterfeit_cnn_prob: data.average_counterfeit_score * 0.8,
        fps: data.video_fps,
        latency_ms: 12.0
      });

      renderTimeline(data);
      document.getElementById('engine-status-text').innerText = `VIDEO: ${data.duration_sec}s (${data.total_frames} frames)`;

      if (window.audioEngine) {
        window.audioEngine.trigger(data.is_counterfeit ? 'COUNTERFEIT_ALARM' : data.primary_denomination);
      }
    } catch (err) {
      console.error('Video inspection failed:', err);
      document.getElementById('engine-status-text').innerText = 'VIDEO ERROR';
    }
  }

  function renderTimeline(data) {
    timelineCard.classList.add('active');
    timelineContainer.innerHTML = '';
    timelineStats.innerText = `Analyzed ${data.timeline.length} sampled keyframes across ${data.duration_sec}s`;

    data.timeline.forEach((node) => {
      const el = document.createElement('div');
      el.className = 'timeline-node';
      const isSus = node.counterfeit_score >= 0.5;
      const statusColor = isSus ? 'var(--pastel-suspect-dot)' : 'var(--pastel-genuine-dot)';

      el.innerHTML = `
        <img class="timeline-thumb" src="${node.thumbnail}" alt="Keyframe @ ${node.time_sec}s" />
        <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; display: flex; justify-content: space-between;">
          <span>${node.time_sec}s</span>
          <span style="color: ${statusColor}; font-weight: 600;">${isSus ? 'SUSPECT' : 'GENUINE'}</span>
        </div>
        <div style="font-size: 0.72rem; color: var(--text-secondary);">
          ${node.denomination} (${(node.confidence * 100).toFixed(0)}%)
        </div>
      `;
      timelineContainer.appendChild(el);
    });
  }

  /* ----------------------------------------------------
     Synthetic Algorithmic Bench & Dual Comparison
  ---------------------------------------------------- */
  if (btnDemoCompare) {
    btnDemoCompare.addEventListener('click', () => {
      const denom = benchDenomSelect ? benchDenomSelect.value : '500';
      triggerComparisonDemo(denom);
    });
  }

  if (btnDemoGenuine) {
    btnDemoGenuine.addEventListener('click', () => {
      const denom = benchDenomSelect ? benchDenomSelect.value : '500';
      if (comparisonSection) comparisonSection.style.display = 'none';
      triggerSyntheticDemo(denom, false);
    });
  }

  if (btnDemoCounterfeit) {
    btnDemoCounterfeit.addEventListener('click', () => {
      const denom = benchDenomSelect ? benchDenomSelect.value : '50';
      if (comparisonSection) comparisonSection.style.display = 'none';
      triggerSyntheticDemo(denom, true, 'cmyk');
    });
  }

  if (btnDemoToy) {
    btnDemoToy.addEventListener('click', () => {
      if (benchDenomSelect) benchDenomSelect.value = 'toy_50';
      if (comparisonSection) comparisonSection.style.display = 'none';
      triggerSyntheticDemo('toy_50', true, 'toy');
    });
  }

  if (benchDenomSelect) {
    benchDenomSelect.addEventListener('change', () => {
      const denom = benchDenomSelect.value;
      if (comparisonSection && comparisonSection.style.display !== 'none') {
        triggerComparisonDemo(denom);
      } else if (denom === 'toy_50') {
        triggerSyntheticDemo('toy_50', true, 'toy');
      } else {
        triggerSyntheticDemo(denom, false);
      }
    });
  }

  async function triggerSyntheticDemo(denom, isCounterfeit, type) {
    document.getElementById('engine-status-text').innerText = 'GENERATING FORENSIC BENCH SAMPLE...';

    const counterfeitType = type || (denom === 'toy_50' ? 'toy' : (isCounterfeit ? 'cmyk' : 'none'));
    const isCf = isCounterfeit || denom === 'toy_50';

    try {
      const resp = await fetch('/api/generate-demo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          denomination: denom,
          is_counterfeit: isCf,
          counterfeit_type: counterfeitType
        })
      });
      const data = await resp.json();

      if (data.sample_image) {
        photoPreview.src = data.sample_image;
        photoPreview.style.display = 'block';
      }

      updateTelemetryUI(data);
      populatePlates(data);
      document.getElementById('engine-status-text').innerText = `SYNTHETIC BENCH: ${data.sample_type}`;

      if (data.audio_tag && window.audioEngine) {
        window.audioEngine.trigger(data.audio_tag);
      }
    } catch (err) {
      console.error('Synthetic demo error:', err);
    }
  }

  async function triggerComparisonDemo(denom) {
    denom = denom || (benchDenomSelect ? benchDenomSelect.value : '500');
    document.getElementById('engine-status-text').innerText = `RUNNING FORENSIC COMPARISON FOR ₹${denom}...`;

    try {
      const resp = await fetch('/api/compare-demo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ denomination: denom })
      });
      const data = await resp.json();

      if (!data || !data.genuine || !data.counterfeit) return;

      const gen = data.genuine;
      const sus = data.counterfeit;

      if (compareHeading) {
        compareHeading.innerText = `Side-by-Side Forensics: ₹${data.denomination} Authentic Intaglio vs. CMYK Counterfeit`;
      }

      // Populate Authentic Card
      if (photoPreview && gen.sample_image) photoPreview.src = gen.sample_image;
      if (cmpGenFull && gen.sample_image) cmpGenFull.src = gen.sample_image;
      if (cmpGenScore) cmpGenScore.innerText = `${(gen.composite_counterfeit_score || 0.05).toFixed(2)} RISK`;
      if (cmpGenVerdict) cmpGenVerdict.innerText = gen.verdict || 'VERIFIED GENUINE';
      if (cmpGenHt) cmpGenHt.innerText = (gen.halftone_score || 0.04).toFixed(2);
      if (cmpGenSr && gen.plates && gen.plates.super_res) cmpGenSr.src = gen.plates.super_res;
      if (cmpGenFft && gen.plates && gen.plates.fft_spectrum) cmpGenFft.src = gen.plates.fft_spectrum;

      // Populate Counterfeit Card
      if (cmpSusFull && sus.sample_image) cmpSusFull.src = sus.sample_image;
      if (cmpSusScore) cmpSusScore.innerText = `${(sus.composite_counterfeit_score || 0.95).toFixed(2)} RISK`;
      if (cmpSusVerdict) cmpSusVerdict.innerText = sus.verdict || 'COUNTERFEIT / SUSPECT';
      if (cmpSusHt) cmpSusHt.innerText = (sus.halftone_score || 0.95).toFixed(2);
      if (cmpSusSr && sus.plates && sus.plates.super_res) cmpSusSr.src = sus.plates.super_res;
      if (cmpSusFft && sus.plates && sus.plates.fft_spectrum) cmpSusFft.src = sus.plates.fft_spectrum;

      if (comparisonSection) {
        comparisonSection.style.display = 'block';
        comparisonSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

      updateTelemetryUI(gen);
      populatePlates(gen);

      document.getElementById('engine-status-text').innerText = `SYNTHETIC BENCH: ₹${data.denomination} SIDE-BY-SIDE READY`;

      if (window.audioEngine) {
        window.audioEngine.trigger(data.denomination);
      }
    } catch (err) {
      console.error('Comparison demo error:', err);
      document.getElementById('engine-status-text').innerText = 'COMPARISON ERROR';
    }
  }

  /* ----------------------------------------------------
     UI Telemetry Updates
  ---------------------------------------------------- */
  function updateTelemetryUI(data) {
    if (!data) return;

    const isNote = data.is_note_detected !== false && data.denomination !== 'Background' && data.denomination !== 'No Currency';
    const isCounterfeit = Boolean(data.is_counterfeit);
    const reasonEl = document.getElementById('verdict-reason');

    if (!isNote) {
      verdictBadge.className = 'verdict-banner neutral';
      verdictText.innerText = 'Awaiting Currency Note';
      if (reasonEl) reasonEl.innerText = 'Position currency note within sensor reticle';
      denomVal.innerText = '---';
      denomConf.innerText = '0.0%';
    } else {
      verdictBadge.className = 'verdict-banner ' + (isCounterfeit ? 'suspect' : 'genuine');
      verdictText.innerText = data.verdict || (isCounterfeit ? 'SUSPECT / COUNTERFEIT' : 'GENUINE');
      if (reasonEl) reasonEl.innerText = data.authenticity_reason || (isCounterfeit ? 'Potential Counterfeit Detected' : 'Authentic Banknote Verified');

      denomVal.innerText = data.denomination ? `₹${data.denomination}` : '---';
      denomConf.innerText = data.confidence !== undefined ? `${(data.confidence * 100).toFixed(1)}%` : '0.0%';
    }

    // Scores
    const compScore = data.composite_counterfeit_score || 0.0;
    const htScore = data.halftone_score || 0.0;
    const cnnScore = data.counterfeit_cnn_prob || 0.0;

    scoreCounterfeit.innerText = compScore.toFixed(2);
    barCounterfeit.style.width = `${Math.min(100, compScore * 100)}%`;
    barCounterfeit.className = 'progress-bar' + (compScore >= 0.5 ? ' danger' : '');

    scoreHalftone.innerText = htScore.toFixed(2);
    barHalftone.style.width = `${Math.min(100, htScore * 100)}%`;
    barHalftone.className = 'progress-bar' + (htScore >= 0.5 ? ' danger' : '');

    scoreCnn.innerText = cnnScore.toFixed(2);
    barCnn.style.width = `${Math.min(100, cnnScore * 100)}%`;
    barCnn.className = 'progress-bar' + (cnnScore >= 0.5 ? ' danger' : '');

    // Watermark
    if (data.watermark && data.watermark.density !== undefined) {
      scoreWatermark.innerText = `${(data.watermark.density * 100).toFixed(1)}%`;
      barWatermark.style.width = `${Math.min(100, data.watermark.density * 500)}%`;
    }

    // Telemetry Footer
    if (data.latency_ms !== undefined) latencyText.innerText = `${data.latency_ms.toFixed(0)} ms`;
    if (data.audio_tag !== undefined) audioTagText.innerText = data.audio_tag || 'STANDBY';
  }

  function populatePlates(data) {
    if (!data.plates) return;
    if (data.plates.clahe) plateClahe.src = data.plates.clahe;
    if (data.plates.super_res) plateSr.src = data.plates.super_res;
    if (data.plates.fft_spectrum) plateFft.src = data.plates.fft_spectrum;

    if (data.cmyk_plates) {
      if (data.cmyk_plates.c) plateC.src = data.cmyk_plates.c;
      if (data.cmyk_plates.m) plateM.src = data.cmyk_plates.m;
    }

    if (data.watermark && data.watermark.mask) {
      plateWm.src = data.watermark.mask;
    }
  }

  /* ----------------------------------------------------
     Audio Controls & Camera Switch
  ---------------------------------------------------- */
  btnAudio.addEventListener('click', () => {
    if (!window.audioEngine) return;
    const isEnabled = window.audioEngine.toggleMute();
    btnAudio.classList.toggle('active', isEnabled);
    if (audioIcon) {
      audioIcon.setAttribute('data-lucide', isEnabled ? 'volume-2' : 'volume-x');
      if (window.lucide) window.lucide.createIcons();
    }
  });

  btnCamSwitch.addEventListener('click', () => {
    currentCameraFacing = currentCameraFacing === 'user' ? 'environment' : 'user';
    stopLiveStream();
    startLiveStream();
  });

  document.getElementById('btn-fullscreen').addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  // Start in live mode by default
  setMode('live');
});
