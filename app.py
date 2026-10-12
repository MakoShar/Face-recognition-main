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
    Flask, render_template, Response, jsonify, request, abort
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
    """Get status of all cameras and the gallery."""
    eng = get_engine()

    return jsonify({
        'cameras': eng.get_all_status(),
        'gallery': eng.get_gallery_info(),
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
    from werkzeug.utils import secure_filename
    from pipeline.enrollment import VideoEnrollmentManager

    safe_name = secure_filename(video_file.filename) or "enrollment.mp4"
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(safe_name)[1], delete=False) as tmp:
        tmp_path = tmp.name
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
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/model/switch', methods=['POST'])
def api_model_switch():
    """Switch active embedding model between glintr100 and adaface."""
    data = request.get_json(silent=True) or {}
    model_name = data.get('model', '').lower()
    if model_name not in ['glintr100', 'adaface']:
        abort(400, description="Invalid model name. Must be 'glintr100' or 'adaface'")

    eng = get_engine()
    try:
        eng.switch_model(model_name)
        return jsonify({
            'status': 'success',
            'active_model': eng.active_model_name,
            'gallery': eng.get_gallery_info()
        })
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/active')
def api_model_active():
    """Get active embedding model name."""
    eng = get_engine()
    return jsonify({
        'active_model': eng.active_model_name,
        'available_models': ['glintr100', 'adaface']
    })


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
