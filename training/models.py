import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from meld_dataset import MELDDataset

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.projection = nn.Linear(768, 256)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        projected_output = self.projection(pooled_output)
        return projected_output
    
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        
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
            nn.Linear(256 + 128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification layer
        self.emo_classifier = nn.Sequential(
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
        
        emotion_output = self.emo_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        
        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }
        
class MultiModelTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset Information:")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        self.optimizer = torch.optim.Adam([
            {'params': self.model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': self.model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': self.model.emo_classifier.parameters(), 'lr': 5e-4},
            {'params': self.model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=2,
        )
        
        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
    def train_epoch(self):
        self.model.train()
        running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}
        
        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)
            
            # calculate losses
            emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss
            
            # backward pass
            total_loss.backward()
            
            # grdadient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()
        
        return {k: v / len(self.train_loader) for k, v in running_loss.items()}
        
    def validate(self):
        self.model.eval()
        val_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_labels = []
        all_emotion_preds = []
        all_sentiment_labels = []
        all_sentiment_preds = []
        
        with torch.inference_mode():
            for batch in self.val_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)
                
                outputs = self.model(text_inputs, video_frames, audio_features)
                
                emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss
                
                all_emotion_preds.extend(torch.argmax(outputs['emotions'], dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(torch.argmax(outputs['sentiments'], dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
                
                val_loss['total'] += total_loss.item()
                val_loss['emotion'] += emotion_loss.item()
                val_loss['sentiment'] += sentiment_loss.item()
        
        avg_loss = {k: v / len(self.val_loader) for k, v in val_loss.items()}
        
if __name__ == "__main__":
    dataset = MELDDataset("dataset/train/train_sent_emo.csv", "dataset/train/train_splits")
    
    sample = dataset[0]
    
    model = MultimodalSentimentModel()
    model.eval()
    
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)
    
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]
    
    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
    }
    
    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }
            
    for i, prob in enumerate(emotion_probs):
        print(f"Emotion: {emotion_map[i]}, Probability: {prob.item():.4f}")
        
    for i, prob in enumerate(sentiment_probs):
        print(f"Sentiment: {sentiment_map[i]}, Probability: {prob.item():.4f}")