import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2


def compose_transformations(cfg, no_augmentations):
    transformations = []

    if cfg.DATALOADER.UPSAMPLE:
        transformations.append(Upsample(cfg.DATALOADER.UPSAMPLE))

    if not no_augmentations:
        crop_size = cfg.AUGMENTATION.CROP_SIZE
        if cfg.AUGMENTATION.IMPORTANCE_CROPPING:
            transformations.append(ImportanceRandomCrop(crop_size))
        else:  # always do cropping
            transformations.append(UniformCrop(crop_size))

        if cfg.AUGMENTATION.RANDOM_FLIP:
            transformations.append(RandomFlip())

        if cfg.AUGMENTATION.RANDOM_ROTATE:
            transformations.append(RandomRotate())

        if cfg.AUGMENTATION.COLORS:
            transformations.append(ColorAugmentation())

    if cfg.AUGMENTATION.HISTOGRAM_EQUALIZATION:
        transformations.append(HistogramEqualization())

    if cfg.AUGMENTATION.NORMALIZE_IMAGENET:
        transformations.append(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Upsample:
    def __init__(self, upsample):
        self.upsample = upsample
    def __call__(self, args):
        img, label = args
        img = cv2.resize(img, (img.shape[1]*self.upsample, img.shape[0]*self.upsample), interpolation=cv2.INTER_CUBIC)
        return img, label


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = np.array(mean, dtype=np.float32)[np.newaxis,np.newaxis, :]
        self.var = np.array(var, dtype=np.float32)[np.newaxis,np.newaxis, :]

    def __call__(self, args):
        img, label = args
        img = (img - self.mean) / self.var
        return img, label


class PadToEven(object):
    def __call__(self, args):
        img, label = args
        img = self.pad_to_even(img)
        label = self.pad_to_even(label)
        return img, label

    @staticmethod
    def pad_to_even(arr: np.ndarray) -> np.ndarray:
        pad_x = 64 - arr.shape[0] % 64 if arr.shape[0] % 64 else 0
        pad_y = 64 - arr.shape[1] % 64 if arr.shape[1] % 64 else 0
        arr = np.pad(arr,
                     [[0, pad_x], [0, pad_y]] + [[0, 0]] * (len(arr.shape) - 2),
                     mode='edge')
        return arr


class Numpy2Torch(object):
    def __call__(self, args):
        img, label = args
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, args):
        img, label = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)

        if vertical_flip:
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

        img = img.copy()
        label = label.copy()

        return img, label


class RandomRotate(object):
    def __call__(self, args):
        img, label = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img, label


class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, img, label):
        img_height, img_width, *_ = img.shape
        y_limit = img_height - self.crop_size
        x_limit = img_width - self.crop_size
        # TODO: crashes when limit 0. this could be handled better
        y = np.random.randint(0, y_limit) if y_limit != 0 else 0
        x = np.random.randint(0, x_limit) if x_limit != 0 else 0

        img_crop = img[y:y + self.crop_size, x:x + self.crop_size, :]
        label_crop = label[y:y + self.crop_size, x:x + self.crop_size, :]
        return img_crop, label_crop

    def __call__(self, args):
        img, label = args
        img, label = self.random_crop(img, label)
        return img, label


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):
        img, label = args

        SAMPLE_SIZE = 5  # an arbitrary number that I came up with
        BALANCING_FACTOR = 20

        random_crops = [self.random_crop(img, label) for i in range(SAMPLE_SIZE)]
        weights = []
        for img_crop, label_crop in random_crops:
            weights.append(label_crop.sum())
        crop_weights = np.array([label_crop.sum() for img_crop, label_crop in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        img_crop_sample, label_crop_sample = random_crops[sample_idx]

        return img_crop_sample, label_crop_sample


class HistogramEqualization(object):
    def __init__(self, clip_limit=3):
        self.clip_limit = clip_limit

    def __call__(self, args):
        img, label = args
        # Convert to 0-255 uint8
        img = (img*255).astype(np.uint8)
        # convert image to LAB color model
        image_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # split the image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(image_lab)

        # apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L channel with the original A and B channel
        merged_channels = cv2.merge((cl, a_channel, b_channel))

        # convert iamge from LAB color model back to RGB color model
        final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

        final_image = final_image.astype(np.float32)/255
        return final_image, label


class ColorAugmentation(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img, label = args
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled, label


