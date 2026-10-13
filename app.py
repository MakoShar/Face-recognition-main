"""
Flask Web Application — Live Face Recognition Dashboard.

Serves a live camera dashboard with MJPEG streaming and
face recognition overlays. Integrates with the recognition
engine without duplicating model instances.
"""

import os
import sys
import time
import logging
import threading
from flask import (
    Flask, render_template, Response, jsonify, request, abort, send_from_directory
)

import config
from pipeline.engine import RecognitionEngine

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Reduce TF/ONNX noise
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('onnxruntime').setLevel(logging.WARNING)

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(config.BASE_DIR, 'templates'),
    static_folder=os.path.join(config.BASE_DIR, 'static'),
)

# ─── Singleton Engine ────────────────────────────────────────────────────────
# Ensures only one engine instance exists regardless of browser connections
engine = None
engine_lock = threading.Lock()


def get_engine():
    """Get or create the singleton RecognitionEngine."""
    global engine
    if engine is not None:
        return engine

    with engine_lock:
        if engine is not None:
            return engine

        logger.info("Creating RecognitionEngine...")
        engine = RecognitionEngine()
        engine.initialize()

        # Add configured cameras
        for cam_id in config.DEFAULT_CAMERA_IDS:
            engine.add_camera(cam_id)

        # Start all cameras
        engine.start_all()
        logger.info("Engine ready and cameras started.")

    return engine


# ─── Allowed Camera IDs (security) ───────────────────────────────────────────
ALLOWED_CAMERA_IDS = set(config.DEFAULT_CAMERA_IDS)


def validate_camera_id(camera_id):
    """Validate camera ID to prevent arbitrary device access."""
    try:
        cam_id = int(camera_id)
    except (ValueError, TypeError):
        abort(400, description="Invalid camera ID")
    if cam_id not in ALLOWED_CAMERA_IDS:
        abort(403, description="Camera ID not allowed")
    return cam_id


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard page."""
    eng = get_engine()
    camera_ids = list(eng.workers.keys())
    gallery_info = eng.get_gallery_info()

    return render_template(
        'dashboard.html',
        camera_ids=camera_ids,
        gallery_info=gallery_info,
        available_models=config.AVAILABLE_MODELS,
        active_model=eng.active_model_name,
    )


@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    """MJPEG video stream for a specific camera."""
    cam_id = validate_camera_id(camera_id)
    eng = get_engine()

    def generate_frames():
        """Generator that yields MJPEG frames."""
        import cv2
        frame_interval = 1.0 / config.MJPEG_MAX_FPS

        while True:
            t_start = time.time()

            frame = eng.get_annotated_frame(cam_id)
            if frame is not None:
                # Encode as JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
                success, buffer = cv2.imencode('.jpg', frame, encode_params)
                if success:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        buffer.tobytes() +
                        b'\r\n'
                    )

            # Rate limit
            elapsed = time.time() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/api/status')
def api_status():
    """Get status of all cameras, gallery, and system hardware metrics."""
    eng = get_engine()

    # System & Process RAM
    import psutil
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    ram_info = {
        'percent': round(vm.percent, 1),
        'used_gb': round(vm.used / (1024**3), 2),
        'total_gb': round(vm.total / (1024**3), 2),
        'available_gb': round(vm.available / (1024**3), 2),
        'process_mb': round(proc.memory_info().rss / (1024**2), 1),
    }

    # GPU VRAM & Utilization
    gpu_info = {
        'available': False,
        'model': 'NVIDIA GeForce RTX 3050',
        'used_mb': 0,
        'total_mb': 4096,
        'percent': 0.0,
        'util_percent': 0.0,
    }
    try:
        import subprocess
        res = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=0.5
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [float(x.strip()) for x in res.stdout.strip().split(',')]
            gpu_info['available'] = True
            gpu_info['used_mb'] = round(parts[0], 0)
            gpu_info['total_mb'] = round(parts[1], 0)
            gpu_info['percent'] = round((parts[0] / parts[1]) * 100, 1)
            gpu_info['util_percent'] = round(parts[2], 0)
    except Exception:
        pass

    return jsonify({
        'cameras': eng.get_all_status(),
        'gallery': eng.get_gallery_info(),
        'system': {
            'ram': ram_info,
            'gpu': gpu_info,
        },
        'timestamp': time.time(),
    })


@app.route('/api/camera/<int:camera_id>/status')
def api_camera_status(camera_id):
    """Get status of a specific camera."""
    cam_id = validate_camera_id(camera_id)
    eng = get_engine()

    status = eng.get_camera_status(cam_id)
    if status is None:
        abort(404, description="Camera not found")

    return jsonify(status)


@app.route('/api/camera/<int:camera_id>/start', methods=['POST'])
def api_camera_start(camera_id):
    """Start a camera's capture and processing."""
    cam_id = validate_camera_id(camera_id)
    eng = get_engine()

    eng.start_camera(cam_id)
    return jsonify({'status': 'started', 'camera_id': cam_id})


@app.route('/api/camera/<int:camera_id>/stop', methods=['POST'])
def api_camera_stop(camera_id):
    """Stop a camera's capture and processing."""
    cam_id = validate_camera_id(camera_id)
    eng = get_engine()

    eng.stop_camera(cam_id)
    return jsonify({'status': 'stopped', 'camera_id': cam_id})


@app.route('/api/gallery/re-enroll', methods=['POST'])
def api_re_enroll():
    """Re-scan the images directory and update the gallery."""
    eng = get_engine()
    count = eng.re_enroll()
    return jsonify({
        'enrolled': count,
        'names': eng.gallery.get_enrolled_names(),
    })


@app.route('/api/gallery')
def api_gallery():
    """Get gallery information."""
    eng = get_engine()
    return jsonify(eng.get_gallery_info())


@app.route('/api/attendance')
def api_attendance():
    """Get recorded attendance events and summary stats."""
    eng = get_engine()
    limit = request.args.get('limit', default=100, type=int)
    return jsonify({
        'events': eng.get_attendance_events(limit=limit),
        'summary': eng.get_attendance_summary(),
    })


@app.route('/api/attendance/clear', methods=['POST'])
def api_attendance_clear():
    """Clear recorded attendance events."""
    eng = get_engine()
    success = eng.clear_attendance()
    return jsonify({
        'status': 'success' if success else 'error',
        'message': 'Attendance log cleared' if success else 'Failed to clear attendance log'
    })


@app.route('/Log/<path:filename>')
def serve_log_screenshot(filename):
    """Serve attendance screenshot image from the Log folder."""
    return send_from_directory(config.LOGS_DIR, filename)


@app.route('/api/enroll/video', methods=['POST'])
def api_enroll_video():
    """
    Enroll an identity using an uploaded video presenting 5 head poses.
    Form data:
      - 'video': Video file (mp4, avi, mov, webm)
      - 'name': Person's display name
    """
    if 'video' not in request.files:
        abort(400, description="Missing 'video' file")
    name = request.form.get('name', '').strip()
    if not name:
        abort(400, description="Missing 'name' field")

    video_file = request.files['video']
    if not video_file.filename:
        abort(400, description="No file selected")

    import tempfile
    import uuid
    from pipeline.enrollment import VideoEnrollmentManager

    tmp_dir = tempfile.gettempdir()
    _, ext = os.path.splitext(video_file.filename)
    if not ext:
        ext = ".mp4"
    tmp_path = os.path.join(tmp_dir, f"enroll_{uuid.uuid4().hex}{ext}")
    video_file.save(tmp_path)

    try:
        eng = get_engine()
        manager = VideoEnrollmentManager(
            target_samples_per_pose=config.MAX_REFS_PER_POSE,
            frame_sample_step=config.FRAME_SAMPLE_INTERVAL
        )
        report = manager.enroll_from_video(
            video_path=tmp_path,
            identity_name=name,
            gallery=eng.gallery,
            detector=eng.detector,
            embedder=eng.embedder,
            pose_estimator=eng.pose_estimator,
            quality_assessor=eng.quality_assessor
        )
        # Save updated gallery
        eng.gallery.save(config.GALLERY_EMBEDDINGS_PATH)
        return jsonify(report)
    except Exception as e:
        logger.exception(f"Video enrollment failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.route('/api/model/switch', methods=['POST'])
def api_model_switch():
    """Switch active embedding model: stops running model first, then starts selected model."""
    data = request.get_json(silent=True) or {}
    model_name = data.get('model', '').lower()
    if model_name not in config.AVAILABLE_MODELS:
        abort(400, description=f"Invalid model name '{model_name}'. Available: {list(config.AVAILABLE_MODELS.keys())}")

    eng = get_engine()
    try:
        switch_res = eng.switch_model(model_name)
        return jsonify({
            'status': 'success',
            'message': switch_res['message'],
            'previous_model': switch_res['previous_model'],
            'active_model': switch_res['active_model'],
            'active_model_name': switch_res['name'],
            'gallery': eng.get_gallery_info()
        })
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/active')
def api_model_active():
    """Get active embedding model name and all available models."""
    eng = get_engine()
    return jsonify({
        'active_model': eng.active_model_name,
        'active_model_name': config.AVAILABLE_MODELS.get(eng.active_model_name, {}).get('name', eng.active_model_name),
        'available_models': config.AVAILABLE_MODELS
    })


# ─── Training State & Endpoints ──────────────────────────────────────────────
training_state = {
    'status': 'idle',  # 'idle', 'running', 'completed', 'error'
    'model': None,
    'model_display': None,
    'progress_percent': 0,
    'current_video': '',
    'current_index': 0,
    'total_videos': 0,
    'message': 'Ready to train',
    'logs': [],
    'result': None,
    'started_at': 0,
    'finished_at': 0,
}
training_lock = threading.Lock()


@app.route('/api/train', methods=['POST'])
def api_train():
    """
    Trigger training / enrollment on all videos in data/raw_videos for a specific model.
    Saves the embeddings separately into gallery/embeddings_<model>.json.
    """
    global training_state

    with training_lock:
        if training_state['status'] == 'running':
            return jsonify({
                'status': 'already_running',
                'message': f"Training is already running for {training_state.get('model_display', 'a model')}",
                'state': training_state
            }), 409

        data = request.get_json(silent=True) or {}
        eng = get_engine()
        target_model = data.get('model', '').lower() or eng.active_model_name

        if target_model not in config.AVAILABLE_MODELS:
            abort(400, description=f"Invalid model '{target_model}'. Available: {list(config.AVAILABLE_MODELS.keys())}")

        model_cfg = config.AVAILABLE_MODELS[target_model]
        training_state = {
            'status': 'running',
            'model': target_model,
            'model_display': model_cfg['name'],
            'progress_percent': 0,
            'current_video': '',
            'current_index': 0,
            'total_videos': 0,
            'message': f"Starting training for {model_cfg['name']}...",
            'logs': [f"[{time.strftime('%H:%M:%S')}] Initialized training for {model_cfg['name']} on {config.GALLERY_VIDEOS_DIR}"],
            'result': None,
            'started_at': time.time(),
            'finished_at': 0,
        }

    def train_worker(model_name):
        global training_state
        try:
            def on_progress(idx, total, filename, name, st, details):
                with training_lock:
                    pct = int((idx / max(1, total)) * 100) if total > 0 else 0
                    training_state['progress_percent'] = pct
                    training_state['current_index'] = idx
                    training_state['total_videos'] = total
                    training_state['current_video'] = filename
                    training_state['message'] = f"[{idx}/{total}] {name}: {details}"
                    log_line = f"[{time.strftime('%H:%M:%S')}] ({idx}/{total}) {name} - {details}"
                    training_state['logs'].append(log_line)
                    if len(training_state['logs']) > 100:
                        training_state['logs'] = training_state['logs'][-100:]

            res = eng.train_model(model_name=model_name, progress_callback=on_progress)
            with training_lock:
                training_state['status'] = 'completed'
                training_state['progress_percent'] = 100
                training_state['finished_at'] = time.time()
                training_state['result'] = res
                training_state['message'] = f"Training completed: {res['identities_enrolled']} identities ({res['total_embeddings']} refs) saved to {os.path.basename(res['cache_file'])}"
                training_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ✓ {training_state['message']}")
        except Exception as e:
            logger.exception(f"Background training error: {e}")
            with training_lock:
                training_state['status'] = 'error'
                training_state['finished_at'] = time.time()
                training_state['message'] = f"Training failed: {str(e)}"
                training_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ✗ Error: {str(e)}")

    t = threading.Thread(target=train_worker, args=(target_model,), daemon=True)
    t.start()

    return jsonify({
        'status': 'started',
        'message': f"Training started for {model_cfg['name']}",
        'model': target_model,
        'model_display': model_cfg['name']
    })


@app.route('/api/train/status')
def api_train_status():
    """Get the live status of the training worker."""
    with training_lock:
        return jsonify(training_state)


# ─── Error Handlers ──────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': str(e.description)}), 400


@app.errorhandler(403)
def forbidden(e):
    return jsonify({'error': str(e.description)}), 403


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': str(e.description)}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Face Recognition Dashboard")
    print("=" * 60)
    print(f"  BlazeFace Model : {config.BLAZEFACE_MODEL_PATH}")
    print(f"  Embedding Model : {config.EMBEDDING_MODEL_PATH}")
    print(f"  Gallery Images  : {config.GALLERY_IMAGES_DIR}")
    print(f"  Cameras         : {config.DEFAULT_CAMERA_IDS}")
    print(f"  Dashboard URL   : http://localhost:{config.FLASK_PORT}")
    print("=" * 60 + "\n")

    # Initialize engine before starting Flask
    get_engine()

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
        use_reloader=False,  # Prevent double model loading
    )
