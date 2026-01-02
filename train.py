# Project imports
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import recall_score
import mlflow

from models.fusion_model import MultimodalTriageModel
from training.dataset import MultimodalDataset


# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["image"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )

            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = batch["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    recall_high = recall_score(
        all_labels,
        all_preds,
        labels=[2],
        average=None
    )

    return recall_high[0]


def main():
    print("Entered main()")

    # 🔧 Force CPU for stability (temporary)
    device = torch.device("cpu")
    torch.set_num_threads(2)

    # Dataset
    dataset = MultimodalDataset(
        csv_path="data/text/symptoms.csv",
        image_root="data/images"
    )

    print("Dataset size:", len(dataset))
    sample = dataset[0]
    print("Sample keys:", sample.keys())

    # Train / Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    # DataLoaders (SAFE for macOS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Model
    model = MultimodalTriageModel(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # MLflow
    mlflow.start_run()
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("batch_size", 8)

    best_recall = 0.0

    # Training loop
    for epoch in range(3):
        print(f"Epoch {epoch+1} started")
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            print("Got batch", i)

            optimizer.zero_grad()

            outputs = model(
                batch["image"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )

            loss = criterion(outputs, batch["label"].to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_recall = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {total_loss:.4f} | "
            f"Val Recall (High Risk): {val_recall:.4f}"
        )

        mlflow.log_metric("train_loss", total_loss, step=epoch)
        mlflow.log_metric("val_recall_high", val_recall, step=epoch)

        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), "best_multimodal_model.pt")
            print("Saved new best model")

    mlflow.end_run()


if __name__ == "__main__":
    main()
