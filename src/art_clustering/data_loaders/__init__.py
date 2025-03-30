import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

class WikiSubsetLoader:
    def __init__(self, root_dir, batch_size=32, img_size=(224, 224), shuffle=False, normalize=False):
        """
        Custom DataLoader for WikiArt dataset, filtering only the selected genres.
        
        :param root_dir: Path to the dataset root folder
        :param selected_genres: List of genre names to include
        :param batch_size: Number of samples per batch
        :param img_size: Target image size for transformations
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.normalize = normalize
        
        # Define transformations
        if self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
            
        # Load dataset using ImageFolder
        self.dataset = ImageFolder(root=root_dir, transform=self.transform)
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=self.shuffle)
    
    def get_loader(self):
        """Returns the DataLoader for the selected subset."""
        return self.dataloader