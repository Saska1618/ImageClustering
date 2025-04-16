import os

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import defaultdict
import random

class WikiSubsetLoaderAgain:
    def __init__(self, root_dir, batch_size=32, img_size=(224, 224), shuffle=False, normalize=False, max_per_class=2000):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.normalize = normalize
        self.max_per_class = max_per_class

        # Transforms
        if self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

        # Load dataset filtering out invalid files
        full_dataset = ImageFolder(root=root_dir, transform=self.transform, is_valid_file=self.is_valid_image)

        # Subsample to max_per_class
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset):
            class_indices[label].append(idx)

        selected_indices = []
        for label, indices in class_indices.items():
            if len(indices) > self.max_per_class:
                indices = random.sample(indices, self.max_per_class)
            selected_indices.extend(indices)

        if self.shuffle:
            random.shuffle(selected_indices)

        self.dataset = Subset(full_dataset, selected_indices)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=self.shuffle)

    def is_valid_image(self, path):
        """Check if the file is a valid image and not a hidden/system file."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        basename = os.path.basename(path)
        return not basename.startswith('.') and any(basename.lower().endswith(ext) for ext in valid_extensions)

    def get_loader(self):
        return self.dataloader