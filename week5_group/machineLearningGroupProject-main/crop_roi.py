import os
import cv2
import numpy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

root_dir = "/Users/wanjiang/Downloads/CMEImages/NoCME_polar"
target_dir = "/Users/wanjiang/Downloads/CMEImages/NoCME_polar_crop"
os.makedirs(target_dir, exist_ok=True)

for filename1 in tqdm(os.listdir(root_dir)):
    filename = os.path.join(root_dir, filename1)
    image = cv2.imread(filename, 0)
    # image = cv2.blur(image, (3, 3))
    # cv2.imshow("image", image)
    # cv2.waitKey()
    image = np.vstack([image, image])
    cme_part = np.zeros((200, 200))
    max_count = -1
    for i in range(image.shape[0]-200):
        sum_of_pixels_under_100_more_than_150 = 0
        image_part = image[i:i+200, :].copy()
        image_part = image_part.astype(np.float32)
        image_part[image_part < 100] = -1
        image_part[image_part > 150] = -1
        count = sum(sum(image_part == -1))
        if count > max_count:
            max_count = count
            cme_part = image[i:i+200, :]
    cv2.imwrite(os.path.join(target_dir, filename1), cme_part)
