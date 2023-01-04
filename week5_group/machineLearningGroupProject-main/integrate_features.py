import os
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

#可调参数，对比实验，pca降的维度
pca_components = 20
def combine_hog_features(positive_path="D:/BaiduNetdiskDownload/image/CMEImages/CME_polar_hog",
                         negative_path="D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar_hog"):

    # (215, 1382400)
    pca = PCA(n_components=pca_components)
    all_samples = []
    for index, filename1 in tqdm(enumerate(os.listdir((positive_path)))):
        filename = os.path.join(positive_path, filename1)
        per_hog = np.load(filename)
        all_samples.append(per_hog.tolist())


    for index, filename1 in tqdm(enumerate(os.listdir((negative_path)))):
        filename = os.path.join(negative_path, filename1)
        per_hog = np.load(filename)
        all_samples.append(per_hog.tolist())
    all_samples = np.array(all_samples)
    downsamples = pca.fit_transform(all_samples)
    # downsamples = all_samples

    labels = np.ones((215 * 2, 1))
    for i in range(215, 430):
        labels[i, 0] = -1
    downsamples = np.column_stack((downsamples, labels))
    np.save("hog_features.npy", downsamples)


def combine_lbp_features(positive_path="D:/BaiduNetdiskDownload/image/CMEImages/CME_polar_lbp",
                         negative_path="D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar_lbp"):

    # (215, 1382400)
    pca = PCA(n_components=pca_components)
    all_samples = []
    for index, filename1 in tqdm(enumerate(os.listdir((positive_path)))):
        filename = os.path.join(positive_path, filename1)
        per_hog = np.load(filename, allow_pickle=True)
        all_samples.append(per_hog.tolist())


    for index, filename1 in tqdm(enumerate(os.listdir((negative_path)))):
        filename = os.path.join(negative_path, filename1)
        per_hog = np.load(filename, allow_pickle=True)
        all_samples.append(per_hog.tolist())
    all_samples = np.array(all_samples)
    # downsamples = all_samples
    downsamples = pca.fit_transform(all_samples)

    labels = np.ones((215 * 2, 1))
    for i in range(215, 430):
        labels[i, 0] = -1
    downsamples = np.column_stack((downsamples, labels))
    np.save("lbp_features.npy", downsamples)

def combine_hist_features(positive_path="D:/BaiduNetdiskDownload/image/CMEImages/CME_polar_hist",
                         negative_path="D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar_hist"):

    # (215, 1382400)
    pca = PCA(n_components=pca_components)
    all_samples = []
    for index, filename1 in tqdm(enumerate(os.listdir((positive_path)))):
        filename = os.path.join(positive_path, filename1)
        per_hog = np.load(filename, allow_pickle=True)
        all_samples.append(per_hog.tolist())


    for index, filename1 in tqdm(enumerate(os.listdir((negative_path)))):
        filename = os.path.join(negative_path, filename1)
        per_hog = np.load(filename, allow_pickle=True)
        all_samples.append(per_hog.tolist())
    all_samples = np.array(all_samples)
    # downsamples = all_samples
    downsamples = pca.fit_transform(all_samples)

    labels = np.ones((215* 2, 1))
    for i in range(215, 430):
        labels[i, 0] = -1
    downsamples = np.column_stack((downsamples, labels))
    np.save("hist_features.npy", downsamples)


combine_hog_features()
combine_lbp_features()
combine_hist_features()
