"""Marker restoration operations. No annotations or image IDs are inputs."""
import cv2
import numpy as np


def color_mask(rgb, threshold):
    a = rgb.astype(np.int16)
    return (a.max(axis=2) - a.min(axis=2)) > threshold


def edit_mask(mask, dilation):
    if not dilation:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=dilation).astype(bool)


def restore(gray, mask, method, radius):
    if not mask.any():
        return gray.copy()
    if mask.all():
        raise ValueError('No observed pixels available for restoration')
    if method == 'harmonic':
        yy, xx = np.nonzero(mask)
        x0, x1 = max(0, xx.min()-2), min(gray.shape[1], xx.max()+3)
        y0, y1 = max(0, yy.min()-2), min(gray.shape[0], yy.max()+3)
        m = mask[y0:y1, x0:x1]
        a = gray[y0:y1, x0:x1].astype(np.float32)
        a[m] = np.median(a[~m])
        for _ in range(400):
            padded = np.pad(a, 1, mode='edge')
            mean = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:])/4
            delta = float(np.max(np.abs(a[m]-mean[m])))
            a[m] = mean[m]
            if delta < 0.01:
                break
        result = gray.copy()
        result[y0:y1, x0:x1][m] = np.clip(np.rint(a[m]), 0, 255).astype(np.uint8)
    else:
        if method not in ('telea', 'ns'):
            raise ValueError(f'Unknown restoration method: {method}')
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        result = cv2.inpaint(gray, mask.astype(np.uint8)*255, radius, flag)
    result[~mask] = gray[~mask]
    return result
