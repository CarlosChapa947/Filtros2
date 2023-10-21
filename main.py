import cv2
import numpy as np
from scipy.signal import convolve2d
import time

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def medianBlur(img, kernel):
    # Use a breakpoint in the code line below to debug your script.
    if kernel % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser un número impar")

    height, width = img.shape[:2]
    pad = kernel // 2
    image_padded = np.pad(img, ((pad, pad)), mode='constant')
    filtered_image = np.zeros_like(img)

    if len(img.shape) == 2:
        print("grey")
        for i in range(height):
            for j in range(width):
                neighborhood = image_padded[i:i + kernel, j:j + kernel]
                median_value = np.median(neighborhood)
                filtered_image[i, j] = median_value
    elif len(img.shape) == 3:
        print("rgb")
        for i in range(height):
            for j in range(width):
                for channel in range(3):  # Iterar sobre canales RGB
                    neighborhood = image_padded[i:i + kernel, j:j + kernel, channel]
                    median_value = np.median(neighborhood)
                    filtered_image[i, j, channel] = median_value

    return filtered_image

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size-1) / 2) ** 2 + (y - (size-1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)



def gaussian_blur(image, kernel, sigma):
    if kernel % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser un número impar")

    gauss_kernel = gaussian_kernel(kernel, sigma)

    filtered_image = np.zeros_like(image, dtype=np.float32)

    if len(image.shape) == 2:
        print("Grey")
        filtered_image[:, :] = convolve2d(image[:, :], gauss_kernel, mode='same', boundary='wrap')
    elif len(image.shape) == 3:
        print("RGB")
        for channel in range(3):  # Iterar sobre canales RGB
            filtered_image[:, :, channel] = convolve2d(image[:, :, channel], gauss_kernel, mode='same', boundary='wrap')

    return filtered_image.astype(np.uint8)

def custom_filter2D(image, kernel):
    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape

    y_strides = m_i - m_k + 1  # Posibles número de pasos en la dirección y
    x_strides = n_i - n_k + 1  # Posibles número de pasos en la dirección x

    img = image
    output_shape = (m_i - m_k + 1, n_i - n_k + 1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0  # Lleva la cuenta de las operaciones de convolución

    output_tmp = output.reshape(
        (output_shape[0] * output_shape[1], output_shape[2])
    )

    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i + m_k, j:j + n_k, c]

                output_tmp[count, c] = np.sum(sub_matrix * kernel)

            count += 1

    output = output_tmp.reshape(output_shape)

    return output.astype(np.uint8)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread("./Images/brain.jpeg", cv2.IMREAD_GRAYSCALE)
    customMedianBlurTimeStart = time.time()
    medianBlur(img, 5)
    customMedianBlurTimeEnd = time.time()
    print(customMedianBlurTimeEnd - customMedianBlurTimeStart)
    cv2MedianBlurTimeStart = time.time()
    cv2.medianBlur(img, 5)
    cv2MedianBlurTimeEnd = time.time()
    print(cv2MedianBlurTimeEnd - cv2MedianBlurTimeStart)
    customGaussBlurStart = time.time()
    gaussImg = gaussian_blur(img, 7, 5.0)
    customGaussBlurEnd = time.time()
    print(customGaussBlurEnd - customGaussBlurStart)
    cv2GaussBlurStart = time.time()
    cv2.GaussianBlur(img, (7, 7), 5)
    cv2GaussBlurEnd = time.time()
    print(cv2GaussBlurEnd - cv2GaussBlurStart)






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
