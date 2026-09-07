import numpy as np 
import cv2
from collections import deque

## helper function for 2d convolution 
def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: str= 'same'):
    H, W = image.shape   # defining the size of image and kernel 
    k_h , k_w = kernel.shape
    if padding.lower() == 'same':
        pad_h =  (k_h- 1) // 2   # defining padding size around the image
        pad_w =  (k_w- 1) // 2
        padded_img = np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),mode='constant', constant_values=0) # perform padding 
    elif padding.lower() == 'valid':
        padded_img = image
    else:
        raise ValueError("Padding must be same or valid")
    # storing the output of the padded image after convolution
    out_h = (padded_img.shape[0] - k_h) // stride + 1  
    out_w = (padded_img.shape[1] - k_w) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            region = padded_img[r_start:r_start+k_h, c_start:c_start+k_w]
            output[i, j] = np.sum(region * kernel)
    return output


def hysteresis_tracking(img_nms: np.ndarray, low_thresh: float, high_thresh:float):
    H, W = img_nms.shape
    strong_i, strong_j = np.where(img_nms >= high_thresh)
    output = np.zeros((H, W), dtype=np.uint8)
    output[strong_i, strong_j] = 255
    queue = deque(zip(strong_i, strong_j))
    while queue:
        ci, cj = queue.popleft()
        for ni in range(max(0, ci- 1), min(H, ci + 2)):
           for nj in range(max(0, cj- 1), min(W, cj + 2)):
                if (img_nms[ni, nj] >= low_thresh) and (output[ni, nj] == 0):
                    output[ni, nj] = 255
                    queue.append((ni, nj))
    return output


## Task 1 : building a convo3d function 

def conv3d(image, kernel_stack , stride = 1, padding='same'):
    H,W,C = image.shape
    k, k_h, k_w, k_c = kernel_stack.shape

    if padding.lower() == 'same':
         pad_h =  (k_h- 1) // 2   # defining padding size around the image
         pad_w =  (k_w- 1) // 2
         padded_img = np.pad(image,((pad_h,pad_h),(pad_w,pad_w),(0,0)),mode='constant') # perform padding
    elif padding.lower() == 'valid':
        padded_img = image 
    else:
        raise ValueError("Padding must be same or valid")
    #storing the output of the padded image after convolution
    out_h = (padded_img.shape[0] - k_h) // stride + 1  
    out_w = (padded_img.shape[1] - k_w) // stride + 1
    output = np.zeros((out_h, out_w,k), dtype=np.float64)
    for k_idx in range(k):
        for i in range(out_h):
            for j in range(out_w):
                r_start = i * stride
                c_start = j * stride
                region = padded_img[r_start:r_start+k_h, c_start:c_start+k_w, :]
                output[i, j,k_idx] = np.sum(region * kernel_stack[k_idx])
    return output

## task 2 : adding gaussian kernel, pooling and unsharpening 
def kernel_gaussian2d(ksize,sigma):
    half = (ksize -1) //2
    x, y = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1))
    kernel = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def unsharping(image,sigma=1.5,alpha=1.2):
    kernel = kernel_gaussian2d(ksize=5,sigma=sigma)
    blurred = conv2d(image,kernel,padding='same')
    detail = image - blurred
    sharpen = image + alpha * detail
    sharpened = np.clip(sharpen, 0, 255)               
    return sharpened

def pooling2d(feature_map,pool_size=(2,2),stride=2, mode='max'):
    H, W = feature_map.shape
    p_h, p_w = pool_size
    out_h = (H - p_h)// stride + 1
    out_w = (W - p_w) //stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            region = feature_map[r_start:r_start+p_h, c_start:c_start+p_w]
            if mode == 'max':
                output[i, j] = np.max(region)
            elif mode == 'average':
                output[i, j] = np.mean(region)
    return output


# task 3 

# NMS 
def non_max_suppression(magnitude, direction):
    H, W =magnitude.shape
    angle =np.degrees(direction) % 180 
    output =np.zeros((H, W))

    for i in range(1, H-1):        
        for j in range(1, W-1):
            a = angle[i, j]
            if (a< 22.5) or (a>= 157.5):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]      # horizontal neighbors
            elif a<67.5:
                n1, n2 =magnitude[i-1, j+1], magnitude[i+1, j-1]  # diagonal neighbors
            elif a <112.5:
                n1, n2 =magnitude[i-1, j], magnitude[i+1, j]      # vertical neighbors
            else:
                n1, n2 =magnitude[i-1, j-1], magnitude[i+1, j+1]  # anti-diagonal neighbors

            if (magnitude[i,j] > n1 and magnitude[i,j] > n2):
                output[i, j] = magnitude[i,j] 
            else:
                output[i,j] = 0

    return output

# full pipeline by integrating all helper functions 
def canny_pipeline(image, sigma=1.0, low_thresh=20, high_thresh=50):
    gaussian_kernel = kernel_gaussian2d(ksize=5, sigma=sigma)
    smoothed = conv2d(image, gaussian_kernel, padding='same')   # Stage 0: reduce noise
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) / 8
    sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) / 8
    gx = conv2d(smoothed, sx, padding='same')
    gy = conv2d(smoothed, sy, padding='same')
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    thinned = non_max_suppression(magnitude, direction)          # Stage 2
    edges = hysteresis_tracking(thinned, low_thresh, high_thresh) # Stage 3

    return edges

    





           
            