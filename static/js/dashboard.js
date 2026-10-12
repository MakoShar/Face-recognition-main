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
                        <p class="text-muted">Add face images to <code>Ppl/Images/</code></p>
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
                    if (detEl) detEl.textContent = `${m.avg_detection_ms.toFixed(1)} ms`;

                    const embEl = document.getElementById(`emb-time-${camId}`);
                    if (embEl) embEl.textContent = `${m.avg_embedding_ms.toFixed(1)} ms`;

                    const faceEl = document.getElementById(`face-count-${camId}`);
                    if (faceEl) faceEl.textContent = m.faces_detected;

                    totalFrames += m.frames_processed;
                    totalLatency += m.avg_total_ms;
                    cameraCount++;
                }
            }
        }

        // Update global stats
        const latencyEl = document.getElementById('avg-latency');
        if (latencyEl && cameraCount > 0) {
            latencyEl.textContent = (totalLatency / cameraCount).toFixed(0);
        }

        const framesEl = document.getElementById('frames-processed');
        if (framesEl) framesEl.textContent = totalFrames.toLocaleString();

        const camerasEl = document.getElementById('active-cameras');
        if (camerasEl) camerasEl.textContent = Object.keys(data.cameras || {}).length;

        // Gallery info
        if (data.gallery) {
            const enrolledEl = document.getElementById('enrolled-count');
            if (enrolledEl) enrolledEl.textContent = (data.gallery.enrolled_names || []).length;
        }

    } catch (error) {
        // Silently ignore polling errors
        console.debug('Status poll failed:', error);
    }
}

// ─── Model Switching ──────────────────────────────────────────────
async function switchModel(modelName) {
    showToast(`Switching to model ${modelName.toUpperCase()}...`, 'info');
    try {
        const response = await fetch('/api/model/switch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: modelName})
        });
        const data = await response.json();
        if (data.status === 'success') {
            showToast(`Active model is now ${modelName.toUpperCase()}`, 'success');
            const statusEl = document.getElementById('status-model-name');
            if (statusEl) statusEl.textContent = `${modelName.toUpperCase()} Active`;
            const footerEl = document.getElementById('footer-active-model');
            if (footerEl) footerEl.textContent = modelName.toUpperCase();
            pollStatus();
        } else {
            showToast(`Model switch failed: ${data.message || 'Error'}`, 'error');
        }
    } catch (err) {
        showToast(`Model switch error: ${err.message}`, 'error');
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

// ─── Initialize ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Start status polling
    pollStatus();
    setInterval(pollStatus, STATUS_POLL_INTERVAL);

    // Initial toast
    showToast('Dashboard connected', 'success', 2500);
});
