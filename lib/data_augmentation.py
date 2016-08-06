#!/usr/bin/env python3

import numpy as np
from lib.config import cfg
from PIL import Image


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype(np.float32)
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select([r == maxc, g == maxc],
                            [bc - gc, 2.0 + rc - bc],
                            default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(arr, hout):
    hsv = rgb_to_hsv(arr)
    hsv[0, ...] += hout  # change hue
    hsv[0, ...] = np.max(np.min(hsv[0, ...], 1), 0)
    rgb = hsv_to_rgb(hsv)
    return rgb


def image_transform(img, crop_x, crop_y, crop_loc=None, color_tint=None):
    """
    Takes numpy.array img
    """

    # Slight translation
    if cfg.TRAIN.RANDOM_CROP and not crop_loc:
        crop_loc = [0]*2
        crop_loc[0] = np.random.randint(0, crop_y)  # corner position row
        crop_loc[1] = np.random.randint(0, crop_x)  # corner position column

    if crop_loc:
        cr, cc = crop_loc
        height, width, channel = img.shape
        img_h = height - crop_y
        img_w = width - crop_x
        img = img[cr:cr+img_h, cc:cc+img_w, :]
        # depth = depth[cr:cr+img_h, cc:cc+img_w]

    if cfg.TRAIN.HUE_CHANGE and not color_tint:
        # color tint
        color_tint = (np.random.rand() - 0.5) * cfg.TRAIN.HUE_RANGE

    if color_tint:
        # Hue change
        img = shift_hue(img, color_tint)

    if cfg.TRAIN.FLIP and np.random.rand() > 0.5:
        img = img[:, ::-1, ...]

    return img


def crop_center(im, new_height, new_width):
    height = im.shape[0]   # Get dimensions
    width = im.shape[1]
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    return im[top:bottom, left:right]


def add_random_background(im, background_img_fns):
    """
    Given PIL.Image object with alpha channel, and a list of background file
    names, return image with a background.
    """
    bg_im = Image.open(np.random.choice(background_img_fns))
    while bg_im.height < im.height or bg_im.width < im.width:
        bg_im = Image.open(np.random.choice(background_img_fns))

    # Randomly crop background to the size of the training image.
    crop_x = np.random.randint(bg_im.width - im.width)
    crop_y = np.random.randint(bg_im.height - im.height)
    blended_im = bg_im.crop((crop_x, crop_y,
                             crop_x + im.width,
                             crop_y + im.height))
    blended_im.paste(im, (0, 0), im)
    return blended_im


def add_random_color_background(im, color_range):
    r = np.random.randint(color_range[0][0], color_range[0][1] + 1)
    g = np.random.randint(color_range[1][0], color_range[1][1] + 1)
    b = np.random.randint(color_range[2][0], color_range[2][1] + 1)

    if isinstance(im, Image.Image):
        im = np.array(im)

    if len(im[0, 0]) < 3:
        raise ValueError('No Alpha Channel in image')

    alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
    im = im[:, :, :3]
    bg_color = np.array([[[r, g, b]]])
    return alpha * bg_color + im


def test(fn):
    import matplotlib.pyplot as plt
    cfg.TRAIN.RANDOM_CROP = True
    cfg.TRAIN.HUE_CHANGE = True
    im = Image.open(fn)
    im = np.asarray(im)[:, :, 0:3]
    imt = image_transform(im, 50, 50)
    plt.imshow(imt)
    plt.show()
