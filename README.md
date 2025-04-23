# EX-05: Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step-1:Load Image:
Read the original image in grayscale using OpenCV.
### Step-2:Add Gaussian Noise:
Generate Gaussian noise using a mean of 0 and a specified standard deviation (e.g., 25).
Add the generated noise to the original image and clip the values to ensure they remain in the valid range (0-255).
### Step-3:Define Filter Kernel:
Create a 3x3 weighted average kernel (e.g., [[1, 2, 1], [2, 4, 2], [1, 2, 1]]), normalizing it by dividing by the sum of the weights (16).
### Step-4:Pad the Image:
Create a padded version of the noisy image to handle border effects during convolution.
### Step-5:Apply Convolution:
Loop through each pixel in the padded image, extract the corresponding region of interest (ROI), and apply the filter by performing an element-wise multiplication and summing the result. Store the filtered values in a new output image.
### Step-6:Display Results:
Use Matplotlib to display the original image, the noisy image, and the filtered image side by side for comparison

## Program:
### Developed By   : Lokesh M
### Register Number: 212223230114


### 1. Smoothing Filters

i) Using Averaging Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('ex-0412.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(12, 4))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/96a2a40a-4fae-48bf-bdfa-b3107734f4b1)
```
gaussian_noise = np.random.normal(0,25, image.shape)
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Gaussian Noise)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/dd61070c-acda-419f-82a0-81f143fcf1d6)

```
filtered_image = np.zeros_like(noisy_image)
height, width = noisy_image.shape
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        filtered_value = np.mean(neighborhood)
        filtered_image[i, j] = filtered_value
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Box Filter 3x3)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/7c8b7419-f104-4969-ba6b-1bfe53678f4b)


ii) Using Weighted Averaging Filter
```
image = cvplt.imshow(image, cmap='gray')
plt.imshow(image,cmap="gray")
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/7f5ef7db-f82b-4851-b63f-4be9dd49a94a)

```
gaussian_noise = np.random.normal(0,25, image.shape)
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Gaussian Noise)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/4e8ca08a-894f-4e3f-8590-39f8ed5ab97c)

```
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16.0  # Normalize the kernel

image_height, image_width = noisy_image.shape
kernel_size = kernel.shape[0]  
pad = kernel_size // 2

padded_image = np.pad(noisy_image, pad, mode='constant', constant_values=0)

filtered_image = np.zeros_like(noisy_image)

for i in range(pad, image_height + pad):
    for j in range(pad, image_width + pad):
        roi = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
        filtered_value = np.sum(roi * kernel)
        filtered_image[i - pad, j - pad] = np.clip(filtered_value, 0, 255)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Weighted Avg)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/9574bd10-48f5-49b7-8783-9a2d3ba16d62)


iii) Using Minimum Filter
```
noisy_image = np.copy(image)
salt_prob = 0.05  
pepper_prob = 0.05  
noisy_image = np.copy(image)
num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255
num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0
```
![image](https://github.com/user-attachments/assets/6f76edbb-7c13-4289-9940-2f471bbb598f)


```
min_filtered_image = np.zeros_like(noisy_image)
max_filtered_image = np.zeros_like(noisy_image)
med_filtered_image = np.zeros_like(noisy_image)
height, width = noisy_image.shape
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        min_filtered_image[i, j] = np.min(neighborhood)
min_filtered_image = np.zeros_like(noisy_image)
plt.imshow(min_filtered_image, cmap='gray')
plt.title('Filtered Image (Min Filter)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/531ed53b-81a9-4d8a-be3d-06cdc760b653)



iv) Using Maximum Filter
```

for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        min_filtered_image[i, j] = np.min(neighborhood)
max_filtered_image = np.zeros_like(noisy_image)
plt.imshow(max_filtered_image, cmap='gray')
plt.title('Filtered Image (Max Filter)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/9d2d3a3c-67f5-47ba-9c59-0483561a3c24)



v) Using Median Filter
```
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        med_filtered_image[i, j] = np.median(neighborhood)
plt.imshow(med_filtered_image, cmap='gray')
plt.title('Filtered Image (Med Filter)')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/0b8dbf4e-2ef5-4bdb-83ff-d70734a25327)


### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```
image = cv2.imread('ex-0412.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/96a2a40a-4fae-48bf-bdfa-b3107734f4b1)
```
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
image_height, image_width = blurred_image.shape
kernel_height, kernel_width = laplacian_kernel.shape
pad_height = kernel_height // 2
pad_width = kernel_width // 2
padded_image = np.pad(blurred_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
```
![image](https://github.com/user-attachments/assets/1fbb6b3c-d45e-4565-bfef-923b1719d5f1)


ii) Using Laplacian Operator
```
laplacian_image = np.zeros_like(blurred_image)
for i in range(image_height):
    for j in range(image_width):
        region = padded_image[i:i + kernel_height, j:j + kernel_width]
        laplacian_value = np.sum(region * laplacian_kernel)
        laplacian_image[i, j] = laplacian_value
laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/57cb9697-7bb1-49f0-b2b0-449dfff1ed09)

```
sharpened_image = cv2.add(image, laplacian_image)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/23ec4360-74f6-4de0-a210-5e8c562e2fac)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
