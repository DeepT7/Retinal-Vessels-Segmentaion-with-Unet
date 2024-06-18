import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion\
                           ,CoarseDropout, RandomBrightnessContrast, RandomGamma, RandomCrop, RandomRotate90, GaussNoise, Blur

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ X = Images and Y = masks """

    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            # 1 Horizontal Flip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            # 2 Vertical Flip
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            # 3 Elastic Deformations
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            # 4 Grid Distortion
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            # 5 Optical Distortion
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            # 6 Random Rotate
            aug = RandomRotate90(p=1)
            augemented = aug(image=x, mask=y)
            x6 = augemented['image']
            y6 = augemented['mask']

            # 7 Contrast
            aug = RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5,p = 1)
            augemented = aug(image=x, mask=y)
            x7 = augemented['image']
            y7 = augemented['mask']

            # 8 Add Gauss Noise
            aug = GaussNoise(var_limit=(30, 70),p = 1)
            augemented = aug(image=x, mask=y)
            x8 = augemented['image']
            y8 = augemented['mask']

            # 9 Blur
            aug = Blur(always_apply=True)
            augemented = aug(image=x, mask=y)
            x9 = augemented["image"]
            y9 = augemented['mask']

            X = [x, x1, x2, x3, x4, x5, x6, x7, x8, x9]
            Y = [y, y1, y2, y3, y4, y5, y6, y7, y8, y9]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(X) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "./DRIVE"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    create_dir("aug_data/train/image")
    create_dir("aug_data/train/mask")
    create_dir("aug_data/test/image")
    create_dir("aug_data/test/mask")

    augment_data(train_x, train_y, "aug_data/train/", augment=True)
    augment_data(test_x, test_y, "aug_data/test/", augment=False)
