import cv2

def resize_normalize(img, normalize_width=1000, interpolate = True):
    w = img.shape[1]
    h = img.shape[0]
    interpolate_mode = cv2.INTER_CUBIC if interpolate else cv2.INTER_NEAREST
    if w>normalize_width:
        resize_ratio = normalize_width / w
        normalize_height = round(h * resize_ratio)
        resize_img = cv2.resize(img, (normalize_width, normalize_height), interpolation=interpolate_mode)
        # cv2.imshow('resize img', resize_img)
        # cv2.waitKey(0)
        return resize_img, resize_ratio
    else:
        return img, 1.0

