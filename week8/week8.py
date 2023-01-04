# %%
# read and plot the data
import numpy as np
from PIL import Image

# vanilla python convolve
def convolve2D(arr, kernel):
    new_arr = np.zeros((arr.shape))
    x,_ = kernel.shape
    # padding
    for i in range(x//2):
        arr = np.insert(arr, 0, values = np.zeros((len(arr[0]))), axis = 0)
        arr = np.insert(arr, len(arr), values = np.zeros((len(arr[0]))), axis = 0)
        arr = np.insert(arr, 0, values = np.zeros((len(arr))), axis = 1)
        arr = np.insert(arr, len(arr[0]), values = np.zeros((len(arr))), axis = 1)
    # convolution
    for a in range(len(new_arr)):
        for b in range(len(new_arr[0])):
            temp = 0
            for i in range(x):
                for j in range(x):
                    temp += kernel[i][j] * arr[a+i][b+j]
            new_arr[a][b] = (temp>255 and 255) or (temp>0 and temp) or 0
    return new_arr

example_arr = np.array([[10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],
                        [10, 10, 10, 0, 0, 0],])
example_kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1],])

print(convolve2D(example_arr, example_kernel))

#%%
# image read
img = Image.open("img_1.jpg")
img_rgb = np.array(img.convert("RGB"))
img_r = img_rgb[ : , : , 0 ] 

kernel_b1 = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1],])
kernel_b2 = np.array([[ 0, -1,  0],
                      [-1,  8, -1],
                      [ 0, -1,  0],])

img_b1 = convolve2D(img_r, kernel_b1)
img_b2 = convolve2D(img_r, kernel_b2)

img_r = Image.fromarray(np.uint8(img_r))
img_r.show()
img_r.save("img_r.jpg")

result_b1 = Image.fromarray(np.uint8(img_b1))
result_b1.show()
result_b1.save("img_b1.jpg")

result_b2 = Image.fromarray(np.uint8(img_b2))
result_b2.show()
result_b2.save("img_b2.jpg")


# %%
