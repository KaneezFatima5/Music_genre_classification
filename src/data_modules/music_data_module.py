from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

class MusicDataModule(LightningDataModule):
    def __init__(self, train_path, test_path, batch_size=10, num_workers=4, validation_split=0.1, classifier="cnn"):
        super().__init__()
        self.test_ds = None
        self.train_ds = None
        self.val_ds = None
        self.classes = None
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split

        mean = [0.485, 0.456, 0.406] if classifier=="transfer_learning" else [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225] if classifier=="transfer_learning" else [0.5, 0.5, 0.5]

        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  #
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(self.train_path, transform=self.train_transforms)
        self.train_ds, self.val_ds = random_split(dataset, [1 - self.validation_split, self.validation_split])
        self.test_ds = datasets.ImageFolder(self.test_path, transform=self.test_transforms)
        self.classes = dataset.classes
        print("Classes:", dataset.classes)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)