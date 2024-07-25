import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset, epochs=10, lr=0.001, device=None):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.train_loader, self.val_loader, self.test_loader = self.dataset.get_loaders()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Trainer initialized on device: {self.device}")

    def train(self):
        best_val_accuracy = 0.0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{len(self.train_loader)}")

            avg_train_loss = train_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}')

            if self.val_loader:
                val_accuracy = self.evaluate(self.val_loader, "Validation")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print("Best model saved.")

        self.evaluate(self.test_loader, "Test")

    def evaluate(self, loader, split_name):
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.model(data)
                loss += F.cross_entropy(outputs, labels, reduction='sum').item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss / total
        accuracy = correct / total
        print(f'{split_name} Loss: {avg_loss:.4f}, {split_name} Accuracy: {accuracy:.4f}')
        return accuracy
