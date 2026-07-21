from torch.utils.data import Dataset
import subprocess
from transformers import AutoTokenizer
import pandas as pd
import os
import cv2
import numpy as np
import torch

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
        
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
    def _load_video_frames(self, video_path):
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
        
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)  # Convert to tensor and change shape to (C, H, W)
    
    def _extract_audio_features(self, video_path):
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
        except Exception as e:
            raise ValueError(f"Error extracting audio features: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{
            row['Utterance_ID']}.mp4"""
        path = os.path.join(self.video_dir, video_filename)
        video_path_exists = os.path.exists(path)
        
        if not video_path_exists:
            raise FileNotFoundError(f"Video file not found: {path}")
        
        text_inputs = self.tokenizer(row['Utterance'], return_tensors="pt", padding='max_length', max_length=128, truncation=True)
        
        # video_frames = self._load_video_frames(path)
        self._extract_audio_features(path)
        
        # print(video_frames)  # Print the shape of the video frames tensor for debugging

if __name__ == "__main__":
    meld = MELDDataset(csv_path="dataset\dev\dev_sent_emo.csv",
                        video_dir="dataset\dev\dev_splits_complete")
    print(meld[0])  # This will print the first item in the dataset and check for video file existence
    