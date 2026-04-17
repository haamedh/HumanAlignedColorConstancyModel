import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import colour


class ColorConstancyDataset(Dataset):
    """
    Loads paired image/ground-truth/segmentation data from two directories
    (e.g. indoor and outdoor) and returns LAB-normalised ground truth.

    Directory structure expected under each path:
        path/img/   - input images (sRGB)
        path/gt/    - ground truth reflectance images (sRGB)
        path/seg/   - segmentation masks
    """
    def __init__(self, indoor_dir, outdoor_dir, transform=None):
        img_paths, gt_paths, seg_paths = [], [], []

        for base_dir in [indoor_dir, outdoor_dir]:
            img_dir = base_dir + "img/"
            gt_dir = base_dir + "gt/"
            seg_dir = base_dir + "seg/"

            for root, _, files in os.walk(img_dir):
                for name in files:
                    img_paths.append(os.path.join(img_dir, name))
            for root, _, files in os.walk(gt_dir):
                for name in files:
                    gt_paths.append(os.path.join(gt_dir, name))
            for root, _, files in os.walk(seg_dir):
                for name in files:
                    seg_paths.append(os.path.join(seg_dir, name))

        self.img_paths = sorted(img_paths)
        self.gt_paths = sorted(gt_paths)
        self.seg_paths = sorted(seg_paths)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_paths[index]).convert('RGB')).astype(np.float32) / 255

        gt = Image.open(self.gt_paths[index]).convert('RGB')
        seg = np.array(Image.open(self.seg_paths[index]).convert('RGB')).astype(np.float32) / 255
        seg = np.where(seg == 0, 1, 0).astype(np.uint8)

        # convert ground truth to normalised LAB
        gt = colour.sRGB_to_XYZ(np.array(gt) / 255)
        gt = colour.XYZ_to_Lab(gt).astype(np.float32)
        gt[:, :, 0] = gt[:, :, 0] / 100
        gt[:, :, 1] = (gt[:, :, 1] + 100) / 200
        gt[:, :, 2] = (gt[:, :, 2] + 100) / 200

        filename = self.img_paths[index][-18:].replace('.png', '')
        filename = filename.replace('/', '_').replace('\\', '_')

        if self.transform:
            image, gt, seg = self.transform(image, gt, seg)
        return index, filename, image, gt, seg
