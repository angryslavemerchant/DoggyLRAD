import time
import numpy as np
import sounddevice as sd
import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MIC_DEVICE      = 1              # device index from sd.query_devices()
SAMPLE_RATE     = 16000          # YAMNet expects 16kHz
BARK_THRESHOLD  = 0.3            # confidence threshold (0.0 - 1.0), tweak this
ESP32_IP        = "192.168.4.1"  # default ESP32 AP address
ESP32_ENDPOINT  = f"http://{ESP32_IP}/bark"
NOTIFY_ESP32    = False          # set True when ESP32 is ready

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter

import csv, urllib.request, os

MODEL_PATH  = "yamnet.tflite"
LABELS_PATH = "yamnet_labels.csv"

if not os.path.exists(MODEL_PATH):
    print("Downloading YAMNet model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite",
        MODEL_PATH
    )
    print("Model downloaded.")

if not os.path.exists(LABELS_PATH):
    print("Downloading labels...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
        LABELS_PATH
    )
    print("Labels downloaded.")

with open(LABELS_PATH, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    LABELS = [row[2] for row in reader]

DOG_KEYWORDS = ["dog", "bark", "bow-wow", "growling", "whimper"]
DOG_INDICES  = [i for i, label in enumerate(LABELS) if any(k in label.lower() for k in DOG_KEYWORDS)]
print(f"Watching classes: {[LABELS[i] for i in DOG_INDICES]}")

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
    audio_float = audio_chunk.flatten().astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], audio_float)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

WINDOW_SAMPLES = 15600  # exactly what YAMNet expects (~0.975s)
HOP_SAMPLES    = 8000   # run inference every ~0.5s (sliding window overlap)

print(f"\n🐶 Bark detector running — listening on device {MIC_DEVICE}")
print(f"   Threshold: {BARK_THRESHOLD}  |  ESP32 notify: {NOTIFY_ESP32}")
print("   Press Ctrl+C to stop\n")

ring_buffer = np.zeros(WINDOW_SAMPLES, dtype="float32")

def audio_callback(indata, frames, time_info, status):
    global ring_buffer
    chunk = indata[:, 0]
    ring_buffer = np.roll(ring_buffer, -len(chunk))
    ring_buffer[-len(chunk):] = chunk

try:
    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=HOP_SAMPLES,
        callback=audio_callback
    ):
        while True:
            time.sleep(HOP_SAMPLES / SAMPLE_RATE)

            scores = run_inference(ring_buffer.reshape(-1, 1))

            # Top 5 classes
            top5_idx = np.argsort(scores)[::-1][:5]
            print("─" * 45)
            for i in top5_idx:
                bar = "█" * int(scores[i] * 20)
                print(f"  {LABELS[i]:<32} {scores[i]:.2f}  {bar}")

            # Bark check
            dog_scores = {LABELS[i]: float(scores[i]) for i in DOG_INDICES}
            top_dog    = max(dog_scores, key=dog_scores.get)
            top_score  = dog_scores[top_dog]

            if top_score >= BARK_THRESHOLD:
                print(f"\n  🔔 BARK DETECTED — {top_dog}: {top_score:.2f}")
                if NOTIFY_ESP32:
                    notify_esp32()

except KeyboardInterrupt:
    print("\nStopped.")