import torch
from EZAI.data import EZDataset
from EZAI.models import get_pretrained_model, CNNModel
from EZAI.trainer import Trainer
from EZAI.utils import save_model, load_model

if __name__ == "__main__":
    # Define the path to the dataset
    data_dir = 'EMNIST_dataset/emnist-digits'

    # Create EZDataset object
    dataset = EZDataset(data_dir, batch_size=64, num_workers=8, pin_memory=True)

    # Get the number of classes using the new getter method
    num_classes = dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")

    # Select a model
    model = CNNModel(num_classes)
    # model = get_pretrained_model('resnet18', num_classes=num_classes, weights=True)

    # Train model
    trainer = Trainer(model, dataset, epochs=10)
    trainer.train()

    # Save model
    save_model(model, 'my_model.pth')

    # Load model
    loaded_model = load_model(get_pretrained_model('resnet18', num_classes=num_classes, weights=False), 'my_model.pth')
