import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

class EZDataset:
    def __init__(self, data_dir, batch_size=32, image_size=(224, 224), test_split=0.2, val_split=0.1, num_workers=1, pin_memory=False, sampling_rate=1):
        print(f"pin_memory: {pin_memory}")
        print(f"sampling_rate: {sampling_rate}")
        print("Initializing EZDataset...")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.test_split = test_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sampling_rate = sampling_rate

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_loader, self.val_loader, self.test_loader, self.classes = self._load_data()
        print("DataLoaders created.")
        print(f"Classes: {self.classes}")

    def _load_data(self):
        if self._is_train_test_split(self.data_dir):
            print(f"Loading train data from {os.path.join(self.data_dir, 'train')}")
            train_loader, val_loader = self._load_split_data(os.path.join(self.data_dir, 'train'))
            train_classes = self._get_classes_from_dataset(train_loader.dataset)
            print(f"Train data loaded with classes: {train_classes}")

            print(f"Loading test data from {os.path.join(self.data_dir, 'test')}")
            test_loader, test_classes = self._load_data_from_dir(os.path.join(self.data_dir, 'test'), train_classes)
            print(f"Test data loaded with classes: {test_classes}")

            if set(train_classes) != set(test_classes):
                raise ValueError("Train and test sets have different classes.")

            return train_loader, val_loader, test_loader, train_classes
        else:
            raise ValueError("Expected directory structure: <data_dir>/train and <data_dir>/test")

    def _is_train_test_split(self, data_dir):
        return os.path.exists(os.path.join(data_dir, 'train')) and os.path.exists(os.path.join(data_dir, 'test'))

    def _load_split_data(self, split_dir, val_split=None):
        val_split = self.val_split if val_split is None else val_split

        full_dataset = datasets.ImageFolder(root=split_dir, transform=self.transform)
        
        # Apply sampling if sampling_rate is specified
        if self.sampling_rate < 1:
            num_samples = int(len(full_dataset) * self.sampling_rate)
            indices = np.random.choice(len(full_dataset), num_samples, replace=False)
            full_dataset = Subset(full_dataset, indices)
        
        num_total = len(full_dataset)
        num_val = int(val_split * num_total)
        num_train = num_total - num_val

        print(f"Split directory: {split_dir}, Total samples: {num_total}, Train samples: {num_train}, Val samples: {num_val}")

        if num_train <= 0 or num_val < 0:
            raise ValueError("Dataset split results in zero samples for one or more splits.")

        train_set, val_set = random_split(full_dataset, [num_train, num_val])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory) if num_train > 0 else None
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory) if num_val > 0 else None

        return train_loader, val_loader

    def _load_data_from_dir(self, dir_path, class_names):
        dataset = datasets.ImageFolder(root=dir_path, transform=self.transform)
        
        # Apply sampling if sampling_rate is specified
        if self.sampling_rate < 1:
            num_samples = int(len(dataset) * self.sampling_rate)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = Subset(dataset, indices)

        # Get classes from the underlying dataset
        dataset_classes = self._get_classes_from_dataset(dataset)

        if set(dataset_classes) != set(class_names):
            raise ValueError("Class names in directory do not match expected class names")

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return data_loader, dataset_classes

    def _get_classes_from_dataset(self, dataset):
        # Recursively get the classes from the underlying dataset
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset.classes

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_num_classes(self):
        return len(self.classes)
