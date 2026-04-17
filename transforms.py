import numpy as np
import random
from torchvision.transforms import functional as F
from skimage.transform import resize, rotate


class ToTensor3(object):
    def __call__(self, image, target, seg):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        seg = F.to_tensor(seg)
        return image, target, seg


class Compose3(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, seg):
        for t in self.transforms:
            image, target, seg = t(image, target, seg)
        return image, target, seg


class RandomCrop3(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    @staticmethod
    def get_params(image, output_size):
        w, h, c = image.shape
        th, tw = output_size
        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        return i, j, th, tw, h, w

    def __call__(self, image, target, seg):
        size = random.randint(100, 400)
        if random.random() < self.prob:
            i, j, th, tw, h, w = self.get_params(image, (size, size))

            image_out = np.zeros((h, w, image.shape[2])).astype('float32')
            target_out = np.zeros((h, w, target.shape[2])).astype('float32')
            seg_out = np.zeros((h, w, seg.shape[2])).astype('float32')

            image = image[i:i + th, j:j + tw, :]
            target = target[i:i + th, j:j + tw, :]
            seg = seg[i:i + th, j:j + tw, :]

            for k in range(image.shape[2]):
                image_out[:, :, k] = resize(image[:, :, k], (h, w), anti_aliasing=True)
            for k in range(target.shape[2]):
                target_out[:, :, k] = resize(target[:, :, k], (h, w), anti_aliasing=True)
            for k in range(seg.shape[2]):
                seg_out[:, :, k] = resize(seg[:, :, k], (h, w), anti_aliasing=True)

            return image_out, target_out, seg_out
        return image, target, seg


class RandomHorizontalFlip3(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, seg):
        if random.random() < self.prob:
            image = np.array(image[:, ::-1, :])
            target = np.array(target[:, ::-1, :])
            seg = np.array(seg[:, ::-1, :])
        return image, target, seg


class RandomVerticalFlip3(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, seg):
        if random.random() < self.prob:
            image = np.array(image[::-1, :, :])
            target = np.array(target[::-1, :, :])
            seg = np.array(seg[::-1, :, :])
        return image, target, seg


class RandomRotationWithCropAndResize3:
    def __init__(self, angle_range=(-30, 30), output_size=(448, 448), prob=0.5):
        self.angle_range = angle_range
        self.output_size = output_size
        self.prob = prob

    def __call__(self, image, target, seg):
        if random.random() < self.prob:
            angle = random.uniform(*self.angle_range)
            original_h, original_w, _ = image.shape

            image_rotated = rotate(image, angle, resize=True, mode='constant', cval=0)
            target_rotated = rotate(target, angle, resize=True, mode='constant', cval=0)
            seg_rotated = rotate(seg, angle, resize=True, mode='constant', cval=0)

            h, w, _ = target_rotated.shape
            new_h = h - original_h
            new_w = w - original_w

            image_cropped = image_rotated[new_h: h - new_h, new_w: w - new_w]
            target_cropped = target_rotated[new_h: h - new_h, new_w: w - new_w]
            seg_cropped = seg_rotated[new_h: h - new_h, new_w: w - new_w]

            image_resized = resize(image_cropped, (original_h, original_w), anti_aliasing=True)
            target_resized = resize(target_cropped, (original_h, original_w), anti_aliasing=True)
            seg_resized = resize(seg_cropped, (original_h, original_w), anti_aliasing=True)

            return image_resized, target_resized, seg_resized
        return image, target, seg
