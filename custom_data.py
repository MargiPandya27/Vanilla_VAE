from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_names = os.listdir(root_dir)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        for class_name in class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        #print(image_path)
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class VAEDataset(LightningDataModule):
    def __init__(self, data_path, train_batch_size=8, val_batch_size=8, test_batch_size=8,
                 patch_size=(32,32), num_workers=0, pin_memory=False):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        # Define the transformation to ensure images are the same size
        transform = transforms.Compose([
            transforms.Resize(self.patch_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Resize to (224, 224)
            transforms.CenterCrop(self.patch_size),  # Crop to ensure exact size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1] range
        ])

        # Setup train and validation datasets
        self.train_dataset = MyDataset(
            root_dir=os.path.join(self.data_dir, 'train'),
            transform=transform
        )
        self.val_dataset = MyDataset(
            root_dir=os.path.join(self.data_dir, 'val'),
            transform=transform
        )
        
        # For the sake of testing, use the validation dataset for testing as well.
        # You can replace this with your actual test set if you have one.
        self.test_dataset = MyDataset(
                root_dir=os.path.join(self.data_dir, 'test'),
                transform=transform
            )


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
