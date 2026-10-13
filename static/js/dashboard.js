/**
 * Face Recognition Dashboard — Client-Side Logic
 *
 * Handles:
 *  - Periodic status polling for metrics and camera health
 *  - Camera start/stop controls
 *  - Gallery re-enrollment
 *  - Feed error/recovery handling
 *  - Toast notifications
 */

// ─── Configuration ─────────────────────────────────────────────────
const STATUS_POLL_INTERVAL = 2000; // ms between status polls
const FEED_RETRY_DELAY = 3000;    // ms before retrying a failed feed

// ─── Toast Notifications ───────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toast-out 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ─── Camera Controls ───────────────────────────────────────────────
async function controlCamera(cameraId, action) {
    try {
        const response = await fetch(`/api/camera/${cameraId}/${action}`, {
            method: 'POST',
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        const data = await response.json();
        showToast(`Camera ${cameraId} ${data.status}`, 'success');

        // Refresh feed if starting
        if (action === 'start') {
            const feed = document.getElementById(`feed-${cameraId}`);
            if (feed) {
                // Force reconnect by resetting src
                setTimeout(() => {
                    feed.src = `/video_feed/${cameraId}?t=${Date.now()}`;
                }, 500);
            }

            const overlay = document.getElementById(`overlay-${cameraId}`);
            if (overlay) overlay.classList.remove('active');
        }

        // Update status immediately
        pollStatus();

    } catch (error) {
        showToast(`Failed to ${action} camera ${cameraId}: ${error.message}`, 'error');
    }
}

// ─── Gallery Re-enrollment ─────────────────────────────────────────
async function reEnroll() {
    const btn = document.getElementById('btn-re-enroll');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Enrolling...';
    }

    try {
        const response = await fetch('/api/gallery/re-enroll', {
            method: 'POST',
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // Update gallery list
        const list = document.getElementById('gallery-list');
        if (list && data.names) {
            list.innerHTML = data.names.map(name => `
                <div class="gallery-item">
                    <div class="gallery-avatar">${name[0]}</div>
                    <span class="gallery-name">${name}</span>
                </div>
            `).join('');

            if (data.names.length === 0) {
                list.innerHTML = `
                    <div class="gallery-empty">
                        <p>No identities enrolled.</p>
                        <p class="text-muted">Add face videos to <code>data/raw_videos/</code></p>
                    </div>
                `;
            }
        }

        // Update count
        const countEl = document.getElementById('enrolled-count');
        if (countEl) countEl.textContent = data.enrolled;

        showToast(`Enrolled ${data.enrolled} identit${data.enrolled === 1 ? 'y' : 'ies'}`, 'success');

    } catch (error) {
        showToast(`Re-enrollment failed: ${error.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"/>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                </svg>
                Re-enroll
            `;
        }
    }
}

// ─── Feed Error Handling ───────────────────────────────────────────
function handleFeedError(img, cameraId) {
    const overlay = document.getElementById(`overlay-${cameraId}`);
    const dot = document.getElementById(`camera-dot-${cameraId}`);
    const statusEl = document.getElementById(`status-${cameraId}`);

    if (overlay) overlay.classList.add('active');
    if (dot) dot.className = 'camera-dot disconnected';
    if (statusEl) {
        statusEl.textContent = 'Disconnected';
        statusEl.style.color = '#e74c6f';
    }

    // Auto-retry after delay
    setTimeout(() => {
        img.src = `/video_feed/${cameraId}?t=${Date.now()}`;
    }, FEED_RETRY_DELAY);
}

function handleFeedLoad(img, cameraId) {
    const overlay = document.getElementById(`overlay-${cameraId}`);
    const dot = document.getElementById(`camera-dot-${cameraId}`);
    const statusEl = document.getElementById(`status-${cameraId}`);

    if (overlay) overlay.classList.remove('active');
    if (dot) dot.className = 'camera-dot';
    if (statusEl) {
        statusEl.textContent = 'Connected';
        statusEl.style.color = '';
    }
}

// ─── Status Polling ────────────────────────────────────────────────
async function pollStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) return;

        const data = await response.json();

        // Update camera-specific stats
        let totalFrames = 0;
        let totalLatency = 0;
        let cameraCount = 0;

        if (data.cameras) {
            for (const [camId, status] of Object.entries(data.cameras)) {
                // Camera dot
                const dot = document.getElementById(`camera-dot-${camId}`);
                if (dot) {
                    if (!status.running) {
                        dot.className = 'camera-dot stopped';
                    } else if (!status.connected) {
                        dot.className = 'camera-dot disconnected';
                    } else {
                        dot.className = 'camera-dot';
                    }
                }

                // Status text
                const statusEl = document.getElementById(`status-${camId}`);
                if (statusEl) {
                    if (!status.running) {
                        statusEl.textContent = 'Stopped';
                        statusEl.style.color = '#64748b';
                    } else if (!status.connected) {
                        statusEl.textContent = 'Reconnecting...';
                        statusEl.style.color = '#f59e0b';
                    } else {
                        statusEl.textContent = 'Connected';
                        statusEl.style.color = '#48c78e';
                    }
                }

                // Error message
                const errEl = document.getElementById(`error-msg-${camId}`);
                if (errEl) {
                    errEl.textContent = status.error || '';
                }

                // Overlay
                const overlay = document.getElementById(`overlay-${camId}`);
                if (overlay) {
                    if (!status.running || !status.connected) {
                        overlay.classList.add('active');
                    }
                    // Don't remove active here — let feed load handler do it
                }

                // Metrics
                if (status.metrics) {
                    const m = status.metrics;

                    const detEl = document.getElementById(`det-time-${camId}`);
                    if (detEl) detEl.textContent = `${(m.latest_detection_ms || m.avg_detection_ms || 0).toFixed(1)} ms`;

                    const embEl = document.getElementById(`emb-time-${camId}`);
                    if (embEl) embEl.textContent = `${(m.latest_embedding_ms || m.avg_embedding_ms || 0).toFixed(1)} ms`;

                    const totEl = document.getElementById(`total-time-${camId}`);
                    if (totEl) totEl.textContent = `${(m.latest_total_ms || m.avg_total_ms || 0).toFixed(1)} ms`;

                    const inFrameEl = document.getElementById(`faces-in-frame-cam-${camId}`);
                    const facesInFrame = m.faces_in_frame !== undefined ? m.faces_in_frame : 0;
                    if (inFrameEl) {
                        inFrameEl.textContent = `${facesInFrame} face${facesInFrame === 1 ? '' : 's'}`;
                        if (facesInFrame > 0) {
                            inFrameEl.classList.add('has-faces');
                        } else {
                            inFrameEl.classList.remove('has-faces');
                        }
                    }

                    // HUD elements on live video
                    const hudFaces = document.getElementById(`hud-faces-${camId}`);
                    if (hudFaces) hudFaces.textContent = facesInFrame;

                    const hudLatency = document.getElementById(`hud-latency-${camId}`);
                    if (hudLatency) hudLatency.textContent = (m.latest_total_ms || m.avg_total_ms || 0).toFixed(0);

                    totalFrames += m.frames_processed;
                    totalLatency += (m.latest_total_ms || m.avg_total_ms || 0);
                    cameraCount++;
                }
            }
        }

        // Live faces in frame across all cameras
        let totalLiveFaces = 0;
        if (data.cameras) {
            for (const status of Object.values(data.cameras)) {
                if (status.metrics && status.metrics.faces_in_frame) {
                    totalLiveFaces += status.metrics.faces_in_frame;
                }
            }
        }

        const facesValEl = document.getElementById('faces-in-frame-val');
        if (facesValEl) facesValEl.textContent = totalLiveFaces;

        const facesBadgeEl = document.getElementById('faces-in-frame-badge');
        if (facesBadgeEl) {
            if (totalLiveFaces > 0) {
                facesBadgeEl.className = 'badge badge-live badge-detected';
                facesBadgeEl.innerHTML = `<span class="pulse-dot active"></span> ${totalLiveFaces} DETECTED`;
            } else {
                facesBadgeEl.className = 'badge badge-live';
                facesBadgeEl.innerHTML = `<span class="pulse-dot"></span> SCANNING`;
            }
        }

        // Update global latency
        const latencyEl = document.getElementById('avg-latency');
        if (latencyEl && cameraCount > 0) {
            latencyEl.textContent = (totalLatency / cameraCount).toFixed(0);
        }

        // Global latency breakdown
        let avgDet = 0;
        let avgEmb = 0;
        if (data.cameras && cameraCount > 0) {
            for (const status of Object.values(data.cameras)) {
                if (status.metrics) {
                    avgDet += (status.metrics.latest_detection_ms || status.metrics.avg_detection_ms || 0);
                    avgEmb += (status.metrics.latest_embedding_ms || status.metrics.avg_embedding_ms || 0);
                }
            }
            avgDet = avgDet / cameraCount;
            avgEmb = avgEmb / cameraCount;
            const breakdownEl = document.getElementById('latency-breakdown');
            if (breakdownEl) {
                breakdownEl.textContent = `Det: ${avgDet.toFixed(1)}ms | Emb: ${avgEmb.toFixed(1)}ms (GPU)`;
            }
        }

        // Update RAM and System metrics
        if (data.system && data.system.ram) {
            const ram = data.system.ram;
            const ramPctEl = document.getElementById('ram-percent');
            if (ramPctEl) ramPctEl.textContent = `${ram.percent}%`;

            const ramDetEl = document.getElementById('ram-details');
            if (ramDetEl) ramDetEl.textContent = `${ram.used_gb} / ${ram.total_gb} GB`;
        }

        // Update GPU metrics & Camera HUD
        if (data.system && data.system.gpu && data.system.gpu.available) {
            const gpu = data.system.gpu;

            // Top metric card: GPU Compute % + VRAM
            const gpuUtilEl = document.getElementById('gpu-util');
            if (gpuUtilEl) gpuUtilEl.textContent = `${gpu.util_percent.toFixed(0)}%`;

            const gpuBadgeEl = document.getElementById('gpu-badge');
            if (gpuBadgeEl) gpuBadgeEl.textContent = `${gpu.used_mb} MB (${gpu.percent}%)`;

            // Update Camera HUD overlay with GPU Compute % and VRAM
            if (data.cameras) {
                for (const camId of Object.keys(data.cameras)) {
                    const hudGpu = document.getElementById(`hud-gpu-${camId}`);
                    if (hudGpu) hudGpu.textContent = `${gpu.util_percent.toFixed(0)}`;

                    const hudVram = document.getElementById(`hud-vram-${camId}`);
                    if (hudVram) hudVram.textContent = `${gpu.used_mb.toFixed(0)}`;
                }
            }
        }

        // Gallery info
        if (data.gallery) {
            const enrolledEl = document.getElementById('enrolled-count');
            if (enrolledEl) enrolledEl.textContent = (data.gallery.enrolled_names || []).length;
        }

    } catch (error) {
        console.debug('Status poll failed:', error);
    }
}

// ─── Model Switching ──────────────────────────────────────────────
let currentActiveModel = document.getElementById('model-select')?.value || 'glintr100';

async function switchModel(modelName) {
    const selectEl = document.getElementById('model-select');
    const selectedOption = selectEl?.options[selectEl.selectedIndex];
    const newModelTitle = selectedOption ? selectedOption.text : modelName.toUpperCase();

    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-model-name');

    // 1. Give clear message that current model is stopping and new model is starting
    showToast(`Stopping current model and starting ${newModelTitle}...`, 'info', 5000);

    if (selectEl) selectEl.disabled = true;
    if (statusDot) statusDot.className = 'status-dot switching';
    if (statusText) statusText.textContent = `Starting ${newModelTitle}...`;

    try {
        const response = await fetch('/api/model/switch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: modelName})
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            currentActiveModel = data.active_model;
            showToast(data.message || `Started ${data.active_model_name} successfully!`, 'success', 5000);

            if (statusDot) statusDot.className = 'status-dot active';
            if (statusText) statusText.textContent = `${(data.active_model_name || modelName).toUpperCase()} Active`;

            const footerEl = document.getElementById('footer-active-model');
            if (footerEl) footerEl.textContent = data.active_model_name || modelName.toUpperCase();

            // Refresh camera feeds to reconnect cleanly with the new model
            setTimeout(() => {
                const feeds = document.querySelectorAll('.camera-feed');
                feeds.forEach(feed => {
                    const srcBase = feed.src.split('?')[0];
                    feed.src = `${srcBase}?t=${Date.now()}`;
                });
            }, 600);

            pollStatus();
        } else {
            throw new Error(data.message || data.error || 'Server error occurred');
        }
    } catch (err) {
        showToast(`Model switch failed: ${err.message}`, 'error', 6000);
        if (selectEl) selectEl.value = currentActiveModel;
        if (statusDot) statusDot.className = 'status-dot active';
        if (statusText) statusText.textContent = `${currentActiveModel.toUpperCase()} Active`;
    } finally {
        if (selectEl) selectEl.disabled = false;
    }
}

// ─── 5-Angle Video Enrollment ──────────────────────────────────────
async function handleVideoEnroll(e) {
    e.preventDefault();
    const nameInput = document.getElementById('enroll-name');
    const fileInput = document.getElementById('enroll-video');
    const btn = document.getElementById('btn-submit-enroll');
    const resultDiv = document.getElementById('enrollment-result');

    if (!nameInput.value || !fileInput.files.length) {
        showToast('Please provide both name and video file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('name', nameInput.value.trim());
    formData.append('video', fileInput.files[0]);

    btn.disabled = true;
    btn.textContent = 'Processing Video...';
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<span style="color:#a5b4fc;">Analyzing frames, detecting poses & assessing quality...</span>';

    try {
        const response = await fetch('/api/enroll/video', {
            method: 'POST',
            body: formData
        });
        const report = await response.json();
        if (response.ok && report.success) {
            showToast(`Enrolled ${report.identity} successfully!`, 'success');
            let poseBadges = Object.entries(report.enrolled_per_pose || {})
                .map(([pose, count]) => `<span class="pose-badge" style="background:${count>0?'rgba(72,199,142,0.2)':'rgba(231,76,111,0.2)'};color:${count>0?'#48c78e':'#e74c6f'};">${pose}: ${count}</span>`)
                .join(' ');
            resultDiv.innerHTML = `
                <div style="color: #48c78e; font-weight:600; margin-bottom: 6px;">
                    ✓ Status: ${report.status} &bull; Total Reference Samples: ${report.total_enrolled_references}
                </div>
                <div>Pose Breakdown: ${poseBadges}</div>
                ${report.missing_poses && report.missing_poses.length > 0 ? `<div style="color:#f59e0b; margin-top:4px;">Missing poses: ${report.missing_poses.join(', ')}</div>` : ''}
            `;
            nameInput.value = '';
            fileInput.value = '';
            pollStatus();
        } else {
            resultDiv.innerHTML = `<span style="color:#e74c6f;">Enrollment failed: ${report.error || 'No usable face frames found'}</span>`;
            showToast(`Enrollment failed: ${report.error || 'Error'}`, 'error');
        }
    } catch (err) {
        resultDiv.innerHTML = `<span style="color:#e74c6f;">Error: ${err.message}</span>`;
        showToast(`Request failed: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Enroll Identity';
    }
}

// ─── Training Modal & Runner ───────────────────────────────────────
let trainPollTimer = null;

function openTrainModal() {
    const modal = document.getElementById('train-modal');
    if (modal) {
        modal.style.display = 'flex';
        // Sync model dropdown with active model
        const select = document.getElementById('train-model-select');
        if (select && currentActiveModel) {
            select.value = currentActiveModel;
        }
        checkTrainingStatus();
    }
}

function closeTrainModal() {
    const modal = document.getElementById('train-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function handleModalBackdropClick(e) {
    if (e.target && e.target.id === 'train-modal') {
        closeTrainModal();
    }
}

async function startTraining() {
    const modelSelect = document.getElementById('train-model-select');
    const startBtn = document.getElementById('btn-start-training');
    const progressSection = document.getElementById('train-progress-section');
    const logContainer = document.getElementById('train-log-container');
    const resultAlert = document.getElementById('train-result-alert');
    const progressBar = document.getElementById('train-progress-bar');
    const percentLabel = document.getElementById('train-percent');
    const stepText = document.getElementById('train-step-text');
    const logBox = document.getElementById('train-log-box');

    const targetModel = modelSelect ? modelSelect.value : currentActiveModel;

    if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerHTML = '<span class="step-spinner"></span> Starting...';
    }

    if (progressSection) progressSection.style.display = 'flex';
    if (logContainer) logContainer.style.display = 'flex';
    if (resultAlert) resultAlert.style.display = 'none';
    if (progressBar) progressBar.style.width = '0%';
    if (percentLabel) percentLabel.textContent = '0%';
    if (stepText) stepText.textContent = `Initializing embedding extraction for ${targetModel.toUpperCase()}...`;
    if (logBox) logBox.innerHTML = `<div style="color:#a5b4fc;">Extracting embeddings from data/raw_videos...</div>`;

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: targetModel })
        });
        const data = await response.json();

        if (!response.ok && data.status !== 'already_running') {
            throw new Error(data.error || data.message || 'Failed to start embedding extraction');
        }

        showToast(`Generating embeddings for ${data.model_display || targetModel}`, 'info');

        // Start polling progress
        if (trainPollTimer) clearInterval(trainPollTimer);
        trainPollTimer = setInterval(pollTrainingProgress, 800);
    } catch (err) {
        showToast(`Embedding error: ${err.message}`, 'error');
        if (resultAlert) {
            resultAlert.className = 'train-result-alert error';
            resultAlert.style.display = 'block';
            resultAlert.textContent = `Failed to generate embeddings: ${err.message}`;
        }
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Generate Embeddings';
        }
    }
}

async function pollTrainingProgress() {
    try {
        const response = await fetch('/api/train/status');
        if (!response.ok) return;
        const state = await response.json();

        const progressBar = document.getElementById('train-progress-bar');
        const percentLabel = document.getElementById('train-percent');
        const stepText = document.getElementById('train-step-text');
        const logBox = document.getElementById('train-log-box');
        const resultAlert = document.getElementById('train-result-alert');
        const startBtn = document.getElementById('btn-start-training');
        const progressSection = document.getElementById('train-progress-section');
        const logContainer = document.getElementById('train-log-container');

        if (state.status === 'running') {
            if (progressSection) progressSection.style.display = 'flex';
            if (logContainer) logContainer.style.display = 'flex';
            if (progressBar) progressBar.style.width = `${state.progress_percent || 0}%`;
            if (percentLabel) percentLabel.textContent = `${state.progress_percent || 0}%`;
            if (stepText) stepText.textContent = state.message || 'Extracting embeddings...';

            if (startBtn) {
                startBtn.disabled = true;
                startBtn.innerHTML = '<span class="step-spinner"></span> Generating Embeddings...';
            }
        }

        // Render logs
        if (logBox && Array.isArray(state.logs)) {
            logBox.innerHTML = state.logs
                .map(l => {
                    let color = '#94a3b8';
                    if (l.includes('✓')) color = '#48c78e';
                    else if (l.includes('✗') || l.includes('Error')) color = '#e74c6f';
                    else if (l.includes('Processing')) color = '#38bdf8';
                    return `<div style="color:${color};">${escapeHtml(l)}</div>`;
                })
                .join('');
            logBox.scrollTop = logBox.scrollHeight;
        }

        if (state.status === 'completed') {
            clearInterval(trainPollTimer);
            trainPollTimer = null;

            if (progressBar) progressBar.style.width = '100%';
            if (percentLabel) percentLabel.textContent = '100%';
            if (stepText) stepText.textContent = state.message;

            if (resultAlert) {
                resultAlert.className = 'train-result-alert success';
                resultAlert.style.display = 'block';
                resultAlert.innerHTML = `
                    <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">✓ Embeddings Generated Successfully!</div>
                    <div>Model: <strong>${escapeHtml(state.model_display || state.model)}</strong></div>
                    <div>Identities Enrolled: <strong>${state.result?.identities_enrolled || 0}</strong> &bull; Total Reference Vectors: <strong>${state.result?.total_embeddings || 0}</strong></div>
                    <div style="font-size:0.8rem; opacity:0.85; margin-top:4px;">Embeddings saved to: <code>${escapeHtml(state.result?.cache_file || '')}</code></div>
                `;
            }

            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Regenerate Embeddings';
            }

            showToast('Embeddings generated! Gallery updated.', 'success');
            // Refresh dashboard gallery
            pollStatus();
        } else if (state.status === 'error') {
            clearInterval(trainPollTimer);
            trainPollTimer = null;

            if (resultAlert) {
                resultAlert.className = 'train-result-alert error';
                resultAlert.style.display = 'block';
                resultAlert.textContent = state.message || 'Embedding extraction encountered an error.';
            }

            if (startBtn) {
                startBtn.disabled = false;
                startBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Retry Generating';
            }
            showToast('Embedding generation encountered an error', 'error');
        }
    } catch (e) {
        console.error('Failed to poll progress:', e);
    }
}

async function checkTrainingStatus() {
    try {
        const response = await fetch('/api/train/status');
        if (!response.ok) return;
        const state = await response.json();
        if (state.status === 'running') {
            if (trainPollTimer) clearInterval(trainPollTimer);
            trainPollTimer = setInterval(pollTrainingProgress, 800);
            pollTrainingProgress();
        }
    } catch (e) {
        // ignore
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ─── Attendance Polling & Controls ─────────────────────────────────
async function pollAttendance() {
    try {
        const response = await fetch('/api/attendance?limit=25');
        if (!response.ok) return;

        const data = await response.json();
        const events = data.events || [];
        const summary = data.summary || {};

        // Update pills
        const countEl = document.getElementById('att-count');
        if (countEl) countEl.textContent = summary.total_events || 0;

        const uniqueEl = document.getElementById('att-unique');
        if (uniqueEl) uniqueEl.textContent = summary.unique_individuals || 0;

        // Update table
        const tbody = document.getElementById('attendance-tbody');
        if (!tbody) return;

        if (events.length === 0) {
            tbody.innerHTML = `
                <tr class="attendance-empty-row">
                    <td colspan="8">
                        <div class="empty-state-content">
                            <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.5">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M12 6v6l4 2"/>
                            </svg>
                            <p>No attendance events logged yet.</p>
                            <span class="text-muted">Requires &ge; 85% confidence and 3 stable tracking frames. Duplicate check-ins suppressed.</span>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        // Format and render recent events
        tbody.innerHTML = events.map(ev => {
            const timeStr = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '--:--:--';
            const confClass = ev.confidence >= 90 ? 'badge-conf-high' : 'badge-conf-med';
            const qualityStr = ev.quality_score ? ev.quality_score.toFixed(1) : '--';
            const poseStr = ev.pose || 'Front';

            return `
                <tr>
                    <td style="font-family: monospace; color: var(--text-secondary);">${timeStr}</td>
                    <td><strong>${escapeHtml(ev.name)}</strong></td>
                    <td><span class="badge-track">#${ev.track_id}</span></td>
                    <td><span class="${confClass}">${(ev.confidence || 0).toFixed(1)}%</span></td>
                    <td style="color: var(--text-secondary);">${qualityStr}</td>
                    <td><span class="mini-pose-badge">${escapeHtml(poseStr)}</span></td>
                    <td><span style="color: var(--text-muted);">Cam ${ev.camera_id}</span></td>
                    <td><span class="badge-logged">Logged</span></td>
                </tr>
            `;
        }).join('');

    } catch (e) {
        console.debug('Attendance poll failed:', e);
    }
}

async function clearAttendance() {
    if (!confirm('Clear all logged attendance records?')) return;

    try {
        const response = await fetch('/api/attendance/clear', { method: 'POST' });
        const data = await response.json();
        if (response.ok && data.status === 'success') {
            showToast('Attendance log cleared', 'success');
            pollAttendance();
        } else {
            showToast('Failed to clear attendance log', 'error');
        }
    } catch (e) {
        showToast(`Error clearing attendance: ${e.message}`, 'error');
    }
}

// ─── Initialize ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Start status polling
    pollStatus();
    setInterval(pollStatus, STATUS_POLL_INTERVAL);

    // Start attendance polling
    pollAttendance();
    setInterval(pollAttendance, 3000);

    // Initial toast
    showToast('Dashboard connected', 'success', 2500);
});
