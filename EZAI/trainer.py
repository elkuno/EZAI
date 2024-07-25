import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def train(self, plot_graph=False):
        best_val_accuracy = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{len(self.train_loader)}")

            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            print(f'Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

            if self.val_loader:
                val_loss, val_accuracy = self.evaluate(self.val_loader, "Validation")
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    print("Best model saved.")

        test_loss, test_accuracy = self.evaluate(self.test_loader, "Test")

        if plot_graph:
            self.plot_training_curve(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy)

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
        return avg_loss, accuracy

    def plot_training_curve(self, train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy):
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(14, 5))

        # Plotting loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        if val_losses:
            plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plotting accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        if val_accuracies:
            plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.scatter(len(epochs) + 1, test_accuracy, label='Test Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training, Validation, and Test Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
