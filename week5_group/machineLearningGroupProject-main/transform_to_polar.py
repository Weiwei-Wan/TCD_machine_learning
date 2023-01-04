import os
import cv2

root_dir = "/Users/wanjiang/Downloads/CMEImages/CME"
target_dir = "/Users/wanjiang/Downloads/CMEImages/NoCME_polar"
os.makedirs(target_dir, exist_ok=True)
index = 0
for filename1 in os.listdir((root_dir)):
    index += 1
    filename = os.path.join(root_dir, filename1)
    img = cv2.imread(filename)
    center = [img.shape[0]//2, img.shape[1]//2]
    polar = cv2.warpPolar(img, dsize = (300, 600), center =  center, maxRadius = center[0],flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    polar = polar[:, 100:]
    print(str(index) + ": " + filename1)
    #cv2.imwrite(os.path.join(target_dir, str(index) + ".jpg"), polar)