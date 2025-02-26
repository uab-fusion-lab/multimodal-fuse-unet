import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RegDBDualModalDataset(Dataset):
    def __init__(self, data_root, index_files, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.indices = []
        for index_file in index_files:
            self.indices.extend(self.load_indices(index_file))
        self.thermal_images = []
        self.visible_images = []
        self.load_images()

    def load_indices(self, index_path):
        index_data = []
        with open(index_path, 'r') as file:
            for line in file:
                thermal_path = line.strip().split()[0]
                visible_path = thermal_path.replace("Thermal", "Visible").replace("_t_", "_v_")
                tmp = thermal_path.split('/')[2].split('_')
                label = self.get_label(tmp[0], tmp[1])
                index_data.append((thermal_path, visible_path, int(label)))
        return index_data

    def get_label(self, gender, position):
        if gender == 'female' and position == 'back':
            return 3
        if gender == 'female' and position == 'front':
            return 2
        if gender == 'male'and position == 'back':
            return 1
        return 0

    def load_images(self):
        for thermal_path, visible_path, label in self.indices:
            thermal_image_path = os.path.join(self.data_root, thermal_path)
            visible_image_path = os.path.join(self.data_root, visible_path)
            self.thermal_images.append((thermal_image_path, label))
            self.visible_images.append((visible_image_path, label))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        thermal_img_path, label = self.thermal_images[idx]
        visible_img_path, _ = self.visible_images[idx]

        thermal_image = Image.open(thermal_img_path)
        visible_image = Image.open(visible_img_path)

        if self.transform:
            thermal_image = self.transform(thermal_image)
            visible_image = self.transform(visible_image)

        return thermal_image, visible_image, label
