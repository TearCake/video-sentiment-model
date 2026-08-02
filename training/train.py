import sys
import subprocess

try:
    import transformers
    import opencv
    import soundfile
except ImportError:
    print("Dependencies missing inside container. Launching forced automated installation...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "transformers==5.14.1",
        "pandas==2.3.3",
        "tqdm==4.69.0",
        "boto3==1.43.61",
        "opencv-python==5.0.0.93",
        "soundfile==0.14.0",
        "tensorboard==2.21.0",
        "scikit-metrics==0.1.1"
    ])
    print("Dependencies successfully configured!")

import argparse
import os
import torch
import torchaudio
from tqdm import tqdm
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
import json
from install_ffmpeg import install_ffmpeg


SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    
    return parser.parse_args()

def main():
    if not install_ffmpeg():
        print("Failed to install ffmpeg.")
        sys.exit(1)

    # install ffmpeg for video processing
    print("Available audio backends :")
    print(str(torchaudio.list_audio_backends()))
    
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # track initial gpu memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Initial GPU memory used: {memory_used:.2f} GB")
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_video_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_video_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_video_dir=os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size
    )
    
    print(f"""Training csv path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video dir path: {os.path.join(args.train_dir, 'train_splits')}""")
    
    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')
    
    metrics_data = {
        'train_losses': [],
        'val_losses': [],
        'epochs': []
    }
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)
        
        metrics_data['train_losses'].append(train_loss)
        metrics_data['val_losses'].append(val_loss)
        metrics_data['epochs'].append(epoch)
        
        # log metrics in sagemaker format
        print(json.dumps({
            "metrics" : [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_recall", "Value": val_metrics["emotion_recall"]},
                {"Name": "validation:emotion_f1_score", "Value": val_metrics["emotion_f1_score"]},
                {"Name": "validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
                {"Name": "validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_recall", "Value": val_metrics["sentiment_recall"]},
                {"Name": "validation:sentiment_f1_score", "Value": val_metrics["sentiment_f1_score"]}
            ]
        }))
        
        # track initial gpu memory if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Initial GPU memory used: {memory_used:.2f} GB")
            
        # save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
    # after training, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data['test_loss'] = test_loss["total"]
    
    print(json.dumps({
        "metrics" : [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
            {"Name": "test:emotion_recall", "Value": test_metrics["emotion_recall"]},
            {"Name": "test:emotion_f1_score", "Value": test_metrics["emotion_f1_score"]},
            {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
            {"Name": "test:sentiment_recall", "Value": test_metrics["sentiment_recall"]},
            {"Name": "test:sentiment_f1_score", "Value": test_metrics["sentiment_f1_score"]}
        ]
    }))
    
if __name__ == "__main__":
    main()