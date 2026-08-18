import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import wave

import boto3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import whisper
from torchvision.models.video import R3D_18_Weights, r3d_18
from transformers import AutoTokenizer, BertModel

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
EMOTION_MAP = {
    0: "anger", 1: "disgust", 2: "fear", 3: "joy",
    4: "neutral", 5: "sadness", 6: "surprise",
}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------------------------------------------------------------------
# Model architecture  (copied verbatim from inference_spec.py)
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(outputs.pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.backbone(x.transpose(1, 2))


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.projection(self.conv_layers(x.squeeze(1)).squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 7),
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3),
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_feat = self.text_encoder(text_inputs["input_ids"], text_inputs["attention_mask"])
        video_feat = self.video_encoder(video_frames)
        audio_feat = self.audio_encoder(audio_features)
        fused = self.fusion_layer(torch.cat([text_feat, video_feat, audio_feat], dim=1))
        return {
            "emotions": self.emotion_classifier(fused),
            "sentiments": self.sentiment_classifier(fused),
        }


# ---------------------------------------------------------------------------
# Helper utilities  (copied verbatim from inference_spec.py)
# ---------------------------------------------------------------------------
def install_ffmpeg():
    if platform.system() != "Linux":
        found = shutil.which("ffmpeg") is not None
        print(f"Non-Linux env — skipping ffmpeg install. ffmpeg on PATH: {found}")
        return True

    print("Installing ffmpeg…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz",
        ])
        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True, text=True,
        )
        ffmpeg_path = result.stdout.strip()
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg binary install error: {e}")

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg check failed: {e}")
        return False


class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224)) / 255.0
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from: {video_path}")
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


def load_wav_as_tensor(audio_path: str):
    """Load a mono 16-bit PCM WAV file into a torch tensor (1, N) normalized to [-1, 1], sample rate."""
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        num_channels = wf.getnchannels()
        raw_bytes = wf.readframes(num_frames)
        
    pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if num_channels > 1:
        pcm16 = pcm16.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
        
    float_audio = pcm16.astype(np.float32) / 32768.0
    waveform = torch.from_numpy(float_audio).unsqueeze(0)
    return waveform, sample_rate


class AudioProcessor:
    def extract_features(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-y", audio_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            waveform, sample_rate = load_wav_as_tensor(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
            )(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, 300 - mel_spec.size(2)))
            else:
                mel_spec = mel_spec[:, :, :300]
            return mel_spec
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg audio extraction failed: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segments(self, video_path, start_time, end_time, temp_dir="/tmp"):
        os.makedirs(temp_dir, exist_ok=True)
        s_time = float(start_time)
        e_time = float(end_time)
        if e_time - s_time < 0.1:
            e_time = s_time + 0.5
        segment_path = os.path.join(temp_dir, f"segment_{s_time}_{e_time}.mp4")
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ss", str(s_time), "-to", str(e_time),
             "-c:v", "libx264", "-c:a", "aac", "-y", segment_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise FileNotFoundError(f"Segment not created: {segment_path}")
        return segment_path


def download_from_s3(s3_uri: str) -> str:
    """Download an S3 object to a local temp file and return its path."""
    s3_uri = s3_uri.rstrip("/")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    
    ext = os.path.splitext(key)[1]
    if not ext or len(ext) > 5:
        ext = ".mp4"
        
    print(f"Downloading s3://{bucket}/{key} ...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        boto3.client("s3").download_file(bucket, key, tmp.name)
        size = os.path.getsize(tmp.name)
        print(f"Downloaded s3://{bucket}/{key} to {tmp.name} (size: {size} bytes)")
        if size < 1000:
            with open(tmp.name, "rb") as f:
                content = f.read()
            print(f"Small file content snippet ({size} bytes): {content}")
        if size == 0:
            raise ValueError(f"Downloaded file from s3://{bucket}/{key} is empty (0 bytes).")
        return tmp.name


# ---------------------------------------------------------------------------
# SageMaker Script-Mode handler functions
# ---------------------------------------------------------------------------

def model_fn(model_dir: str):
    """Called once when the container starts. Returns the loaded model dict."""
    if not install_ffmpeg():
        raise RuntimeError("FFmpeg installation failed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultimodalSentimentModel().to(device)
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pth not found in {model_dir}")

    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transcriber = whisper.load_model(
        "base", device="cpu" if device.type == "cpu" else "cuda"
    )

    print("model_fn complete — model is ready.")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "transcriber": transcriber,
        "device": device,
    }


def input_fn(request_body, request_content_type: str):
    """Deserialise the raw request body into a Python dict."""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: dict, model_dict: dict):
    """Run inference. Logic mirrors VideoSentimentInferenceSpec.invoke()."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    video_path = input_data.get("video_path") if isinstance(input_data, dict) else None
    if not video_path:
        raise ValueError("Request JSON must contain a 'video_path' key (S3 URI or local path).")

    local_path = download_from_s3(video_path) if video_path.startswith("s3://") else video_path

    result = model_dict["transcriber"].transcribe(local_path, word_timestamps=True)
    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result["segments"]:
        segment_path = None
        try:
            segment_path = utterance_processor.extract_segments(
                local_path, segment["start"], segment["end"]
            )
            video_frames = utterance_processor.video_processor.process_video(segment_path)
            audio_features = utterance_processor.audio_processor.extract_features(segment_path)

            text_inputs = tokenizer(
                segment["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_prob = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_prob = torch.softmax(outputs["sentiments"], dim=1)[0]
                emotion_values, emotion_indices = torch.topk(emotion_prob, k=3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_prob, k=3)

            emotion_list = [
                {"label": EMOTION_MAP[idx.item()], "confidence": prob.item(), "probability": prob.item()}
                for idx, prob in zip(emotion_indices, emotion_values)
            ]
            sentiment_list = [
                {"label": SENTIMENT_MAP[idx.item()], "confidence": prob.item(), "probability": prob.item()}
                for idx, prob in zip(sentiment_indices, sentiment_values)
            ]

            predictions.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "emotions": emotion_list,
                "sentiments": sentiment_list,
                "emotion": emotion_list,
                "sentiment": sentiment_list,
            })
        except Exception as e:
            print(f"Error processing segment {segment}: {e}")
        finally:
            if segment_path and os.path.exists(segment_path):
                os.remove(segment_path)

    if video_path.startswith("s3://") and os.path.exists(local_path):
        os.remove(local_path)

    return {"utterances": predictions}


def output_fn(prediction: dict, response_content_type: str):
    """Serialise the prediction dict to JSON."""
    return json.dumps(prediction), "application/json"
