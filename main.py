import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
from model import ThreeDCNN, LSTM, MasterRCNN, DCSASSDataset, train, evaluate


def load_dataset_from_structure(dataset_root):
    video_paths = []
    labels = []

    for class_dir in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_dir)
        if not os.path.isdir(class_path):
            continue

        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(".mp4"):
                    video_paths.append(os.path.join(root, file))
                    labels.append(class_dir)  # Class name is folder name

    return video_paths, labels


def main():
    # Configuration
    config = {
        "dataset_root": os.path.normpath(os.path.join(os.getcwd(), "dataset", "DCSASS Dataset")),
        "learning_rate": 0.001,
        "num_epochs": 10,
        "batch_size": 8,
        "sequence_length": 16,
        "output_dir": "output"
    }

    os.makedirs(config["output_dir"], exist_ok=True)

    print(f"\nLoading videos from: {config['dataset_root']}")
    video_paths, labels = load_dataset_from_structure(config["dataset_root"])

    # Filter out classes with only one video
    label_counts = Counter(labels)
    valid_indices = [i for i, label in enumerate(labels) if label_counts[label] > 1]
    video_paths = [video_paths[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    assert len(video_paths) == len(labels), "Mismatch between video and label counts"
    print(f"\nDataset loaded: {len(video_paths)} videos with {len(set(labels))} classes")

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = DCSASSDataset(
        video_paths=train_paths,
        labels=train_labels,
        transform=train_transform,
        sequence_length=config["sequence_length"]
    )

    val_dataset = DCSASSDataset(
        video_paths=val_paths,
        labels=val_labels,
        transform=val_transform,
        sequence_length=config["sequence_length"]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = MasterRCNN(num_classes=len(set(labels))).to(device)

    # Optional: Dynamically set Linear layer size
    if hasattr(model, "infer_fc_input_size"):
        model.infer_fc_input_size((3, config["sequence_length"], 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")

        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, os.path.join(config["output_dir"], "best_model.pth"))
            print(f"Saved best model with accuracy: {val_acc:.4f}")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

