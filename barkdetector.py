import time
import threading
import numpy as np
import sounddevice as sd
import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MIC_DEVICE      = 0              # device index from sd.query_devices()
SAMPLE_RATE     = 16000          # YAMNet expects 16kHz
TOP_N           = 3              # bark detected if a dog class appears in top N
ESP32_IP        = "192.168.4.1"  # default ESP32 AP address
ESP32_ENDPOINT  = f"http://{ESP32_IP}/on"
NOTIFY_ESP32    = True          # set True when ESP32 is ready

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
DOG_INDICES  = set(i for i, label in enumerate(LABELS) if any(k in label.lower() for k in DOG_KEYWORDS))
print(f"Watching classes: {[LABELS[i] for i in DOG_INDICES]}")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ─── NOTIFY ───────────────────────────────────────────────────────────────────

def notify_esp32():
    try:
        requests.get(ESP32_ENDPOINT, timeout=2)
        print("  -> ESP32 notified!")
    except Exception as e:
        print(f"  -> ESP32 notify failed: {e}")

# ─── INFERENCE ────────────────────────────────────────────────────────────────

def run_inference(audio_chunk):
    audio_float = audio_chunk.flatten().astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], audio_float)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ─── QUIT LISTENER ────────────────────────────────────────────────────────────

running = True

def listen_for_quit():
    global running
    while True:
        if input().strip().lower() == 'q':
            print("\nStopping...")
            running = False
            break

quit_thread = threading.Thread(target=listen_for_quit, daemon=True)
quit_thread.start()

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

WINDOW_SAMPLES = 15600
HOP_SAMPLES    = 8000

print(f"\nBark detector running -- listening on device {MIC_DEVICE}")
print(f"Detection: dog class in top {TOP_N}  |  ESP32 notify: {NOTIFY_ESP32}")
print("Press q + enter to stop\n")

ring_buffer = np.zeros(WINDOW_SAMPLES, dtype="float32")

def audio_callback(indata, frames, time_info, status):
    global ring_buffer
    chunk = indata[:, 0]
    ring_buffer = np.roll(ring_buffer, -len(chunk))
    ring_buffer[-len(chunk):] = chunk

with sd.InputStream(
    device=MIC_DEVICE,
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=HOP_SAMPLES,
    callback=audio_callback
):
    while running:
        time.sleep(HOP_SAMPLES / SAMPLE_RATE)

        scores = run_inference(ring_buffer.reshape(-1, 1))
        top_indices = np.argsort(scores)[::-1]

        print("---")
        for i in top_indices[:5]:
            marker = " <dog>" if i in DOG_INDICES else ""
            bar = "#" * int(scores[i] * 20)
            print(f"  {LABELS[i]:<32} {scores[i]:.2f}  {bar}{marker}")

        top_n_set = set(top_indices[:TOP_N])
        triggered = top_n_set & DOG_INDICES
        if triggered:
            best = max(triggered, key=lambda i: scores[i])
            print(f"\n  !! BARK DETECTED -- {LABELS[best]} (rank {list(top_indices).index(best) + 1})")
            if NOTIFY_ESP32:
                notify_esp32()

print("Stopped.")