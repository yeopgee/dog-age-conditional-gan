"""AFAC image dataset with young and senior labels."""
import os
from PIL import Image
from torch.utils.data import Dataset

class AFACDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.samples = []
        self.label_map = {'Young': 0, 'Senior': 1}
        self.missing_files = []

        # Keep only labels with an image available on disk.
        with open(txt_path, 'r') as f:
            for line in f:
                filename, label = line.strip().split('\t')

                img_path = os.path.join(img_dir, filename)
                if os.path.exists(img_path):
                    self.samples.append((filename, label))
                else:
                    self.missing_files.append(filename)

        print(f"✅ 총 로드된 샘플 수: {len(self.samples)}")
        if self.missing_files:
            print(f"⚠️ 누락된 이미지 파일 수: {len(self.missing_files)}")
            print(f"예시 누락 파일명: {self.missing_files[:5]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_map[label]
        return image, label_idx
