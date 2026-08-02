import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models.video import r3d_18, R3D_18_Weights

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.projection = nn.Linear(768, 128)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        projected_output = self.projection(pooled_output)
        return projected_output
    
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
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # (batch, frames, channels, height, width) -> (batch, channels, frames, height, width)
        x = x.transpose(1, 2)
        return self.backbone(x)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        for param in self.conv_layers.parameters():
            param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dimension if present
        features = self.conv_layers(x)
        return self.projection(features.squeeze(-1))  # Flatten the output for the linear layer
    
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification layer
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # anger, disgust, fear, happiness, sadness, surprise, neutral
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # negative, neutral, positive
        )
        
    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'], text_inputs['attention_mask']
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)
        
        # Concatenate features
        combined_features = torch.cat([text_features, video_features, audio_features], dim=1) # (batch_size, 128 * 3)
        
        fused_features = self.fusion_layer(combined_features)
        
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        
        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }