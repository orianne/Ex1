
from builtins import len
import numpy as np
import imageio as iio
import skimage
from matplotlib import pyplot as plt
from skimage.color import rgb2gray



def is_grayscale_image(img):
    # (5,5) -- Grayscale Image Shape Option #1
    # (5,5,1) -- Grayscale Image Shape Option #2 (Is it tough?)
    if len(img.shape) == 2 or img.shape[2] == 1:
        return True
    return False


def read_image(filename, representation):
    img = iio.imread(filename)
    img = img.astype(np.float64)
    if is_grayscale_image(img) and representation == 1:  # if img is gray
        return img / 255
    elif is_grayscale_image(img) is False and representation == 2:
        return img / 255
    elif is_grayscale_image(img) is False and representation == 1:
        im_g = rgb2gray(img)
        return im_g/255


def imdisplay(filename, representation):
    img = read_image(filename, representation)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    imYIQ = imRGB.dot(mat.transpose())
    return imYIQ


def yiq2rgb(imYIQ):
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    mat = np.linalg.inv(mat)
    return imYIQ.dot(mat.transpose())


def histogram_equalize_image(im_orig):
    im_orig_255 = np.round((im_orig * 255)).astype(np.uint8)
    hist_orig, bins = np.histogram(im_orig_255, bins=256)
    hist_cum = np.cumsum(hist_orig)
    first = (hist_cum != 0).argmax(axis=0)
    if(hist_cum[255] - hist_cum[first] == 0):
        return [im_orig, hist_orig, hist_orig]
    hist_nor = (hist_cum - hist_cum[first]) / (hist_cum[255] - hist_cum[first])
    hist_nor = hist_nor * 255
    look_up = np.round(hist_nor).astype(np.uint8)
    im_eq = np.array(list(map(lambda i: look_up[i], im_orig_255)))
    hist_eq, bins = np.histogram(im_eq, bins=256)
    return [im_eq/255, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    if is_grayscale_image(im_orig):
        return histogram_equalize_image(im_orig)
    else:
        yiq = rgb2yiq(im_orig)
        # 0 is the Y channel
        y_eq, hist_orig, hist_eq = histogram_equalize_image(yiq[..., 0])
        yiq[..., 0] = y_eq
        im_eq = yiq2rgb(yiq)
        return [im_eq, hist_orig, hist_eq]



def quantize(im_orig, n_quant, n_iter):
    if is_grayscale_image(im_orig):
        return quantize_image(im_orig, n_quant, n_iter)
    else:
        yiq = rgb2yiq(im_orig)
        # 0 is the Y channel
        _im_quant, error = quantize_image(yiq[..., 0], n_quant, n_iter)
        yiq[..., 0] = _im_quant
        im_quant = yiq2rgb(yiq)
        return [im_quant, error]


def quantize_image(im_orig, n_quant, n_iter):
    im_orig_255 = np.round((im_orig * 255)).astype(np.uint8)
    hist, bins = np.histogram(im_orig_255, bins=256)

    # z initialize
    z = np.arange(n_quant + 1)
    im_cumsum = np.cumsum(hist)
    section = im_cumsum[-1] / n_quant
    z_equal = np.linspace(section, im_cumsum[-1], n_quant)
    for i in range(n_quant):
        z[i + 1] = np.argmin(im_cumsum < z_equal[i])

    # q initialize
    q = np.arange(n_quant)
    for x in range(n_quant):
        q[x] = (z[x] + z[x+1] + 1)/2

    error = np.array([0] * n_iter).astype(np.uint8)

    for x in range(n_iter):
        error_x = 0
        # Store previous z values to check convergence
        z_prev = z.copy()
        # as the equation from class
        for i in range(n_quant):
            start, end = z[i], z[i + 1] + 1
            top = sum(hist[start:end] * np.arange(start, end))
            bottom = sum(hist[start:end])
            q[i] = top / bottom
            error_x += np.sum((hist[start:end] - q[i]) ** 2)

        # Calculate z by updated q
        for i in range(1, n_quant):
            z[i] = (q[i - 1] + q[i]) / 2

        # Stop iterating if z did not change
        if np.array_equal(z, z_prev):
            break
        error[x] = error_x

    # Build quantization lookup table
    look_up = np.arange(256)
    for i in range(n_quant):
        start, end = z[i], z[i + 1]
        look_up[start:end + 1] = q[i]

    # Map
    im_quant = np.array(list(map(lambda a: look_up[a],im_orig_255))).astype(np.uint8)
    im_quant = im_quant / 256

    return [im_quant, error]


# def main():
#     x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
#     grad = np.tile(x, (256, 1))
#     # imdisplay('camel.jpg', 1)
#     # img = iio.imread('camel.jpg')
#     # img = img.astype(np.float64)
#     # img /= 255
#     # yiq_img = rgb2yiq(img)
#     # rgb_img = yiq2rgb(yiq_img)
#     # rgb_img2 = skimage.color.yiq2rgb(yiq_img)
#     # print('isclose: ', np.isclose(rgb_img, rgb_img2))
#     # print('isclose: ', np.isclose(img, rgb_img))
#     im_orig = read_image('camel.jpg', 1)
#     # img = iio.imread('camel.jpg')
#     # im_eq, hist_orig, hist_eq = histogram_equalize(img/255)
#
#     # im_orig = read_image('camel.jpg', 1);
#     img , error = quantize(im_orig , 4 ,10)
#     plt.imshow(img, cmap=plt.cm.gray)
#     plt.show()
# main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
