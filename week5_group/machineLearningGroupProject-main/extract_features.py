import os
import numpy as np
import cv2
from skimage import feature
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import pyplot as plt

radius = 2
n_point = radius * 8
def lbp_texture(image):
    # 使用skimage LBP方法提取图像的纹理特征
    lbp = feature.local_binary_pattern(image,n_point,radius,'default')
    # 统计图像直方图256维
    max_bins = int(lbp.max() + 1)
    # hist size:256
    lbp_feature, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return lbp_feature

def extract_lbp_feature(root_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename1 in tqdm(os.listdir((root_dir))):
        filename = os.path.join(root_dir, filename1)
        img = cv2.imread(filename, 0)
        lbp_feature = lbp_texture(img)
        np.save(os.path.join(target_dir, filename1[:-4] + '.npy'), lbp_feature)

def extract_hog_features(root_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename1 in tqdm(os.listdir((root_dir))):
        filename = os.path.join(root_dir, filename1)
        img = cv2.imread(filename, 0)

        image_features = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),
            block_norm='L2-Hys')
        print(image_features.shape)

        np.save(os.path.join(target_dir, filename1[:-4] + '.npy'), image_features)

def extract_hist(root_dir, target_dir, show_dir):
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(show_dir, exist_ok=True)
    for filename1 in tqdm(os.listdir((root_dir))):
        plt.figure()
        filename = os.path.join(root_dir, filename1)
        img = cv2.imread(filename, 0)
        img = [img]
        hist = cv2.calcHist(img, [0], None, [256], [0, 256])
        # print(img.shape)
        # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        hist = hist[:, 0]
        plt.plot(hist, 'r')
        plt.savefig(os.path.join(show_dir, filename1[:-4] + ".png"))
        #np.save(os.path.join(target_dir, filename1[:-4] + '.npy'), hist)

# extract_hog_features("/Users/wanjiang/Downloads/CMEImages/NoCME_polar_crop", "/Users/wanjiang/Downloads/CMEImages/NoCME_polar_hog", )
# extract_hog_features("/Users/wanjiang/Downloads/CMEImages/CME_polar_crop",   "/Users/wanjiang/Downloads/CMEImages/CME_polar_hog")
# extract_lbp_feature("/Users/wanjiang/Downloads/CMEImages/NoCME_polar_crop", "/Users/wanjiang/Downloads/CMEImages/NoCME_polar_lbp")
# extract_lbp_feature("/Users/wanjiang/Downloads/CMEImages/CME_polar_crop", "/Users/wanjiang/Downloads/CMEImages/CME_polar_lbp")
extract_hist("/Users/wanjiang/Downloads/CMEImages/NoCME_polar_crop", "/Users/wanjiang/Downloads/CMEImages/NoCME_polar_hist", "/Users/wanjiang/Downloads/CMEImages/NoCME_polar_hist_show")
extract_hist("/Users/wanjiang/Downloads/CMEImages/CME_polar_crop", "/Users/wanjiang/Downloads/CMEImages/CME_polar_hist", "/Users/wanjiang/Downloads/CMEImages/CME_polar_hist_show")







