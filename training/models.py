import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models.video import r3d_18, R3D_18_Weights
from meld_dataset import MELDDataset
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

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
        
def compute_class_weights(dataset):
    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0
    total = len(dataset)
    
    print("Computing class distributions...")
    for i in range(total):
        sample = dataset[i]
        
        if sample is None:
            skipped += 1
            continue
        
        emotion_label = sample['emotion_label']
        sentiment_label = sample['sentiment_label']
        
        if isinstance(emotion_label, torch.Tensor):
            emotion_label = emotion_label.item()
        if isinstance(sentiment_label, torch.Tensor):
            sentiment_label = sentiment_label.item()
        
        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1
        
    valid = total - skipped
    
    if valid == 0:
        raise ValueError("Error: Zero valid samples found. Check if your video data files exist inside the container.")
    
    print(f"Total samples: {total}, Skipped samples: {skipped}, Valid samples: {valid}")
    
    print("\nClass distributions:")
    print("Emotions : ")
    emotion_map = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
    for i, count in enumerate(emotion_counts):
        print(f"{emotion_map[i]}: {count/valid:.2f}")
    
    print("\nSentiments : ")
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    for i, count in enumerate(sentiment_counts):
        print(f"{sentiment_map[i]}: {count/valid:.2f}")
        
    # calculate class weights
    emotion_weights = 1.0 / (emotion_counts + 1e-6)
    sentiment_weights = 1.0 / (sentiment_counts + 1e-6)
    
    # normalize weights
    emotion_weights = emotion_weights / emotion_weights.sum()
    sentiment_weights = sentiment_weights / sentiment_weights.sum()
    
    return emotion_weights, sentiment_weights

class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset Information:")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S") # Jul30_17-36-56
        base_dir = '/opt/ml/output/tensorboard_logs' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f'{base_dir}/run_{timestamp}'
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        
        self.optimizer = torch.optim.Adam([
            {'params': self.model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': self.model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': self.model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': self.model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=2,
        )
        
        self.current_train_losses = None
        
        # calculate class weights
        print("\nCalculating class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(train_loader.dataset)
        
        device = next(model.parameters()).device
        
        self.emotion_weights = emotion_weights.to(device)
        self.sentiment_weights = sentiment_weights.to(device)
        
        print(f"Emotion class weights on device: {self.emotion_weights.device}")
        print(f"Sentiment class weights on device: {self.sentiment_weights.device}")
        
        self.emotion_criterion = nn.CrossEntropyLoss(
            weight=self.emotion_weights,
            label_smoothing=0.05
        )
        self.sentiment_criterion = nn.CrossEntropyLoss(
            weight=self.sentiment_weights,
            label_smoothing=0.05
        )

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else:
            self.writer.add_scalar('loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar('loss/total/val', losses['total'], self.global_step)
            
            self.writer.add_scalar('loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar('loss/emotion/val', losses['emotion'], self.global_step)
            
            self.writer.add_scalar('loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar('loss/sentiment/val', losses['sentiment'], self.global_step)
        
        if metrics:
            self.writer.add_scalar(f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step)
            self.writer.add_scalar(f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step)
            self.writer.add_scalar(f'{phase}/emotion_recall', metrics['emotion_recall'], self.global_step)
            self.writer.add_scalar(f'{phase}/emotion_f1', metrics['emotion_f1'], self.global_step)
            
            self.writer.add_scalar(f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step)
            self.writer.add_scalar(f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step)
            self.writer.add_scalar(f'{phase}/sentiment_recall', metrics['sentiment_recall'], self.global_step)
            self.writer.add_scalar(f'{phase}/sentiment_f1', metrics['sentiment_f1'], self.global_step)

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
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
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
            
            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })
            
            self.global_step += 1
        
        return {k: v / len(self.train_loader) for k, v in running_loss.items()}
        
    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
        all_emotion_labels = []
        all_emotion_preds = []
        all_sentiment_labels = []
        all_sentiment_preds = []
        
        with torch.inference_mode():
            for batch in self.data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)
                
                outputs = self.model(text_inputs, video_frames, audio_features)
                
                emotion_loss = self.emotion_criterion(outputs['emotions'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiments'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss
                
                all_emotion_preds.extend(torch.argmax(outputs['emotions'], dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(torch.argmax(outputs['sentiments'], dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
                
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()
        
        avg_loss = {k: v / len(data_loader) for k, v in losses.items()}
        
        # Calculate metrics
        emotion_precision = precision_score(all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        emotion_recall = recall_score(all_emotion_labels, all_emotion_preds, average='weighted')
        emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average='weighted')
        
        sentiment_precision = precision_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        sentiment_recall = recall_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
        sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='weighted')
        
        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'emotion_recall': emotion_recall,
            'emotion_f1': emotion_f1,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy,
            'sentiment_recall': sentiment_recall,
            'sentiment_f1': sentiment_f1
        }, phase=phase)
        
        if phase == "val":
            self.scheduler.step(avg_loss['total'])
        
        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'emotion_recall': emotion_recall,
            'emotion_f1': emotion_f1,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy,
            'sentiment_recall': sentiment_recall,
            'sentiment_f1': sentiment_f1
        }

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