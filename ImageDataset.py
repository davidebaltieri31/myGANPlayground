import torch
import cv2 as cv
import numpy as np
import torch.utils.data as data
import os
from imgaug import augmenters as iaa
from random import randrange
from torchvision import transforms
import torch.utils.data as torchdata

class ImageDatasetFolder(data.Dataset):
    def has_file_allowed_extension(self, filename, extensions):
        return filename.lower().endswith(extensions)

    def make_dataset(self, path, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        _path = os.path.expanduser(path)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(path, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    _path = os.path.join(root, fname)
                    if self.has_file_allowed_extension(_path, extensions):
                        item = (_path, class_to_idx[target])
                        images.append(item)
        return images

    def __init__(self, path, start_size, do_rgb=False, preload=False, augmentations=None, center_crop=False, nonpreserving_scale=False):
        self.do_center_crop = center_crop
        self.non_preserving_scale = nonpreserving_scale
        self.extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        self.classes = [d.name for d in os.scandir(path) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = self.make_dataset(path, self.class_to_idx, self.extensions)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + path + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))
        self.path = path
        self.size = start_size
        if augmentations is None:
            self.augmenter = iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=True),
                        iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255)),
                        iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255), per_channel=True),
                        iaa.AdditivePoissonNoise(lam=(0, 16)),
                        iaa.AdditivePoissonNoise(lam=(0, 16), per_channel=True)
                    ])
        else:
            self.augmenter = augmentations

        self.rgb = do_rgb
        self.images = {}
        self.preloaded = False
        if preload:
            self.preloaded = True
            for s, (img_path, target) in enumerate(self.samples):
                if self.rgb:
                    img = cv.imread(img_path, cv.IMREAD_COLOR)
                    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
                else:
                    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                self.images[img_path] = img

        if self.rgb:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])

    def random_crop(self, image):
        h = image.shape[0]
        w = image.shape[1]
        if w == h:
            return image
        if w>h:
            x = randrange(0, w-h)
            y = 0
            w = h
        else:
            x = 0
            y = randrange(0, h-w)
            h = w
        if self.rgb:
            image = image[y:y+h, x:x+w, :]
        else:
            image = image[y:y+h, x:x+w]
        return image

    def center_crop(self, image):
        h = image.shape[0]
        w = image.shape[1]
        if w == h:
            return image
        if w>h:
            x = (w-h)//2
            y = 0
            w = h
        else:
            x = 0
            y = (h-w)//2
            h = w
        if self.rgb:
            image = image[y:y+h, x:x+w, :]
        else:
            image = image[y:y+h, x:x+w]
        return image

    def __getitem__(self, index):
        if self.preloaded:
            img_path, target = self.samples[index]
            img = self.images[img_path]
        else:
            img_path, target = self.samples[index]
            #print(img_path)
            if self.rgb:
                img = cv.imread(img_path, cv.IMREAD_COLOR)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            else:
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = self.augmenter.augment_image(img)
        if self.non_preserving_scale is False:
            if self.do_center_crop is True:
                img = self.center_crop(img)
            else:
                img = self.random_crop(img)
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_AREA)
        if self.rgb is False:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        return fmt_str

    def set_size(self, new_size):
        self.size = new_size

#testing
if __name__ == '__main__':
    datasets_location = "D:/Development/Datasets/"
    dataset_artist_dir = "Ritratti/"
    dataset = ImageDatasetFolder("D:\\Development\\Datasets\\Ritratti\\", 224, True, False, center_crop=True, nonpreserving_scale=False)
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        image = data[0]
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image + 1.0)/2.0
        cv.imshow("image",image)
        cv.waitKey(25)
