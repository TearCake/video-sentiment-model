import subprocess
from transformers import AutoTokenizer
import torchaudio
import whisper
from models import TextEncoder, VideoEncoder, MultimodalSentimentModel
import torch
import os
import cv2
import numpy as np
import sys
import json
import boto3
import tempfile

EMOTION_MAP = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
SENTIMENT_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

def install_ffmpeg():
    print("Starting installation of ffmpeg...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("ffmpeg installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing ffmpeg: {e}")
    
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])
        
        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])
        
        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()
        
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])
        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])
        
        print("ffmpeg binary installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing ffmpeg binary: {e}")
        
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("ffmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error occurred while checking ffmpeg version: {e}")
        return False

class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Video file not found: {video_path}")
            
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read frame from video: {video_path}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
            
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # Resize frame to 224x224 and normalize
                frames.append(frame)
                
            
        except Exception as e:
            print(f"Error loading video frames: {e}")
        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames were loaded from video: {video_path}")
        
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
        
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

class AudioProcessor:
    def extract_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')
        
        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
            
            mel_spec = mel_spectrogram(waveform)
            
            # normalize the mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]
                
            return mel_spec  # Return the mel spectrogram tensor
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed to extract audio: {e}")
                
        except Exception as e:
            raise ValueError(f"Audio Error: {e}")
        
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
    
    def extract_segments(self, video_path, start_time, end_time, temp_dir="/tmp"):
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")
        
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise FileNotFoundError(f"Segment file not created: {segment_path}")
        
        return segment_path

def download_from_s3(s3_uri):
    s3_client = boto3.client('s3')
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        s3_uri = input_data['video_path']
        local_path = download_from_s3(s3_uri)
        return {'video_path': local_path}
    raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")

def model_fn(model_dir):
    # load model and tokenizer
    if not install_ffmpeg():
        raise RuntimeError("FFmpeg installation failed. Please ensure FFmpeg is installed and accessible.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained("bert-base-uncased"),
        'transcriber': whisper.load_model(
            "base",
            device= "cpu" if device.type == "cpu" else "cuda"
        ),
        'device': device
    }
    
def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']
    
    result = model_dict['transcriber'].transcribe(video_path, word_timestamps=True)
    
    utterance_processor = VideoUtteranceProcessor()
    predictions = []
    
    for segment in result['segments']:
        try:
            segment_path = utterance_processor.extract_segments(
                video_path, segment['start'], segment['end']
            )
            
            video_frames = utterance_processor.video_processor.process_video(segment_path)
            audio_features = utterance_processor.audio_processor.extract_features(segment_path)
            text_inputs = tokenizer(
                segment['text'],
                return_tensors="pt",
                padding='max_length',
                max_length=128,
                truncation=True
            )
            
            text_inputs = { k: v.to(device) for k, v in text_inputs.items() }
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)
            
            # get predictions
            with torch.no_grad():
                outputs = model(text_inputs, video_frames, audio_features)
                emotion_prob = torch.softmax(outputs['emotions'], dim=1)[0]
                sentiment_prob = torch.softmax(outputs['sentiments'], dim=1)[0]
                
                emotion_values, emotion_indices = torch.topk(emotion_prob, k=3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_prob, k=3)
                
            predictions.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'text': segment['text'],
                'emotion': [
                    {'label': EMOTION_MAP[idx.item()], 'probability': prob.item()} for idx, prob in zip(emotion_indices, emotion_values)
                ],
                'sentiment': [
                    {'label': SENTIMENT_MAP[idx.item()], 'probability': prob.item()} for idx, prob in zip(sentiment_indices, sentiment_values)
                ]
            })
            
        except Exception as e:
            print(f"Error processing segment {segment}: {e}")
        
        finally:
            if os.path.exists(segment_path):
                os.remove(segment_path)
    return {"utterances": predictions}
    
"""
def process_local_video(video_path, model_dir="model"):
    model_dict = model_fn(model_dir)
    input_data = {'video_path': video_path}
    
    predictions = predict_fn(input_data, model_dict)
    
    for utterance in predictions['utterances']:
        print("\nUtterance:")
        print(f"Start Time: {utterance['start_time']:.2f}s, End Time: {utterance['end_time']:.2f}s")
        print(f"Text: {utterance['text']}")
        print("Top 3 Emotions:")
        for emotion in utterance['emotion']:
            print(f"  - {emotion['label']}: {emotion['probability']:.2f}")
        print("Top 3 Sentiments:")
        for sentiment in utterance['sentiment']:
            print(f"  - {sentiment['label']}: {sentiment['probability']:.2f}")
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "dia2_utt3.mp4")
    process_local_video(video_path)
"""