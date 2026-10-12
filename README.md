# Face Recognition System

Real-time, multi-person face recognition using **BlazeFace** for detection and **GlintR100 (ArcFace)** for identity recognition, with a Flask web dashboard.

## Architecture

```
Camera → Frame Sampler → BlazeFace Detection → Face Crop →
GlintR100 Embedding → Gallery Matching → Identity Output → Flask Dashboard
```

## Quick Start

### 1. Install Dependencies

Most dependencies are already installed. To verify:

```bash
pip install flask opencv-python numpy onnxruntime tensorflow pillow scikit-learn
```

### 2. Enroll Identities

Place face images in the `Ppl/Images/` directory:

```
Ppl/Images/
├── Anoop_Verma.jpeg      → Enrolled as "Anoop Verma"
├── Swetabh_Tripathy.jpeg → Enrolled as "Swetabh Tripathy"
└── New_Person.jpg        → Enrolled as "New Person"
```

**Naming convention:** Filename (without extension) becomes the identity name. Underscores are replaced with spaces.

### 3. Launch the Dashboard

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### 4. Test with a Static Image

```bash
python -c "
from pipeline.engine import RecognitionEngine
import config, cv2

engine = RecognitionEngine()
engine.initialize()

results, annotated = engine.process_test_image('Ppl/Test/Test_Image.jpeg')
for r in results:
    print(f'{r.name} (similarity: {r.similarity:.3f}) at {r.bbox}')

cv2.imwrite('test_output.jpg', annotated)
print('Saved: test_output.jpg')
"
```

## Project Structure

```
Face-recognition/
├── app.py                  # Flask application entry point
├── config.py               # All configurable settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── Model/
│   ├── Face-Detection/
│   │   └── blaze_face_full_range_sparse.tflite  # BlazeFace (192×192)
│   └── antelopev2/
│       └── glintr100.onnx                       # ArcFace ResNet100 (112×112 → 512-d)
│
├── Ppl/
│   ├── Images/             # Enrollment photos (auto-enrolled at startup)
│   └── Test/               # Test images for offline validation
│
├── pipeline/
│   ├── detector.py         # BlazeFace TFLite wrapper
│   ├── preprocessor.py     # Face crop + normalization
│   ├── embedder.py         # GlintR100 ONNX embedding
│   ├── gallery.py          # Identity enrollment + matching
│   ├── tracker.py          # IoU-based frame-to-frame tracking
│   ├── camera.py           # Threaded camera capture
│   └── engine.py           # Full pipeline orchestrator
│
├── gallery/
│   └── embeddings.json     # Saved gallery (auto-generated)
│
├── static/                 # Web assets (CSS, JS)
└── templates/              # Jinja2 HTML templates
```

## Models

| Model | File | Input | Output | Purpose |
|---|---|---|---|---|
| BlazeFace Full Range | `blaze_face_full_range_sparse.tflite` | `[1, 192, 192, 3]` float32 | Bounding boxes + confidence | Face detection |
| GlintR100 (ArcFace) | `glintr100.onnx` | `[N, 3, 112, 112]` float32 | `[N, 512]` embeddings | Face recognition |

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `COSINE_SIMILARITY_THRESHOLD` | 0.45 | Match acceptance threshold |
| `BLAZEFACE_CONFIDENCE_THRESHOLD` | 0.65 | Face detection confidence |
| `FRAME_SAMPLE_INTERVAL` | 5 | Process every Nth frame |
| `TRACKER_ENABLED` | True | Enable face tracking |
| `MJPEG_QUALITY` | 80 | Browser stream quality |
| `DEFAULT_CAMERA_IDS` | [0] | Camera device IDs |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard page |
| `/video_feed/<cam_id>` | GET | MJPEG video stream |
| `/api/status` | GET | System status + metrics |
| `/api/camera/<id>/start` | POST | Start a camera |
| `/api/camera/<id>/stop` | POST | Stop a camera |
| `/api/gallery/re-enroll` | POST | Re-scan images and update gallery |
| `/api/gallery` | GET | Gallery information |

## Adding Cameras

Edit `config.py`:

```python
DEFAULT_CAMERA_IDS = [0, 1, 2]  # Add more camera device IDs
```

## Performance Notes

- **BlazeFace detection:** ~3-5 ms per frame on CPU
- **GlintR100 embedding:** ~30-80 ms per face on CPU (larger model, higher accuracy)
- **Frame sampling:** Every 5th frame from 30 FPS = ~6 inferences/second
- **Tracking:** Reduces redundant recognition by caching identity across frames
