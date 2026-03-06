import time
import numpy as np
import sounddevice as sd
import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MIC_DEVICE      = 1           # device index from sd.query_devices()
SAMPLE_RATE     = 15600       # YAMNet expects 16kHz
CLIP_DURATION   = 1           # seconds per inference chunk
BARK_THRESHOLD  = 0.1         # confidence threshold (0.0 - 1.0), tweak this
ESP32_IP        = "192.168.4.1"  # default ESP32 AP address
ESP32_ENDPOINT  = f"http://{ESP32_IP}/bark"
NOTIFY_ESP32    = False       # set True when ESP32 is ready

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter

import csv, urllib.request, os

MODEL_PATH  = "yamnet.tflite"
LABELS_PATH = "yamnet_labels.csv"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading YAMNet model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite",
        MODEL_PATH
    )
    print("Model downloaded.")

# Download labels if not present
if not os.path.exists(LABELS_PATH):
    print("Downloading labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
        LABELS_PATH
    )
    print("Labels downloaded.")

# Load labels
with open(LABELS_PATH, newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    LABELS = [row[2] for row in reader]

# Dog-related label indices
DOG_KEYWORDS = ["dog", "bark", "bow-wow", "growling", "whimper"]
DOG_INDICES  = [i for i, label in enumerate(LABELS) if any(k in label.lower() for k in DOG_KEYWORDS)]
print(f"Watching classes: {[LABELS[i] for i in DOG_INDICES]}")

# Load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ─── NOTIFY ───────────────────────────────────────────────────────────────────

def notify_esp32():
    try:
        requests.get(ESP32_ENDPOINT, timeout=2)
        print("  → ESP32 notified!")
    except Exception as e:
        print(f"  → ESP32 notify failed: {e}")

# ─── INFERENCE ────────────────────────────────────────────────────────────────

def run_inference(audio_chunk):
    # YAMNet expects float32 mono audio at 16kHz
    audio_float = audio_chunk.flatten().astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], audio_float)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    return scores

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

print(f"\n🐶 Bark detector running — listening on device {MIC_DEVICE}")
print(f"   Threshold: {BARK_THRESHOLD}  |  ESP32 notify: {NOTIFY_ESP32}")
print("   Press Ctrl+C to stop\n")

try:
    while True:
        # Record a chunk of audio
        audio = sd.rec(
            int(CLIP_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=MIC_DEVICE
        )
        sd.wait()

        scores = run_inference(audio)

        # Check dog-related classes
        dog_scores = {LABELS[i]: float(scores[i]) for i in DOG_INDICES}
        top_dog    = max(dog_scores, key=dog_scores.get)
        top_score  = dog_scores[top_dog]

        if top_score >= BARK_THRESHOLD:
            print(f"🔔 BARK DETECTED — {top_dog}: {top_score:.2f}")
            if NOTIFY_ESP32:
                notify_esp32()
        else:
            # Uncomment to see live scores even when quiet:
            # print(f"   quiet — top dog class: {top_dog}: {top_score:.2f}")
            pass

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped.")