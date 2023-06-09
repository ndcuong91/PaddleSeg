import os
import numpy as np
import cv2
from onnxruntime import InferenceSession
from random import random
import time

from PIL import Image

def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im

def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.
    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im, label=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.
        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = np.asarray(Image.open(label))
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = np.transpose(im, (2, 0, 1))
        return (im, label)


class Resize:
    """
    Resize an image.
    Args:
        target_size (list|tuple, optional): The target size of image. Default: (512, 512).
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".
    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                        format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                    .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, label=None):

        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = resize(label, self.target_size, cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


class Normalize:
    """
    Normalize an image.
    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].
    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean,
                           (list, tuple)) and isinstance(self.std,
                                                         (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, label=None):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)

        if label is None:
            return (im,)
        else:
            return (im, label)


class Segment_onnx:
    def __init__(self, model_path, input_size=(512, 512)):
        self.sess = InferenceSession(model_path)
        self.transform = Compose([Resize(target_size=input_size), Normalize()])
        self.input_size = input_size

    def inference(self, img):
        '''
        :param img: opencv img
        :return: mask img
        '''
        input = self.transform(img)[0]
        input = input[np.newaxis, ...]

        output_mask = self.sess.run(output_names=None,
                                    input_feed={self.sess.get_inputs()[0].name: input})
        return output_mask


def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG','jpeg','JPEG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def order_points(pts, vertical=False):
    '''
    Sắp xếp lại các điểm phục vụ cho transformation, trong trường hợp thẻ dọc thì sẽ sắp xếp khác thẻ ngang
    :param pts:
    :param vertical: thẻ dọc
    :return:
    '''
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    if vertical:
        rect[3] = pts[np.argmin(s)]
        rect[1] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(diff)]
        rect[2] = pts[np.argmax(diff)]
    else:
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
    return rect


def find_quadrilateral(img_ori, mask, sw, sh, debug=False, max_object=1):
    '''
    Tìm tứ giác bao quanh mask đã được segment
    :param img_ori: anh goc
    :param mask:
    :param sw: scale width
    :param sh: scale height
    :param debug:
    :param max_object: Số tứ giác tối đa detect được
    :return:
    '''
    list_quads = []
    img = mask
    size_min = min(img.shape[0], img.shape[1])
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, he = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, he = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    list_contours = [f for f in contours]
    list_contours.sort(key=lambda x: x.size, reverse=True)
    # list_contours = list_contours[:max_object]

    # img_contours = np.zeros(img_ori.shape)
    # # draw the contours on the empty image
    # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    #
    # # Convert image to BGR (just for drawing a green rectangle on it).
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #
    # cv2.drawContours(bgr_img, [box], 0, (0, 255, 0), 2)
    #
    # # Show images for debugging
    # cv2.imshow('bgr_img', bgr_img)
    # cv2.waitKey()

    thickness = int(img_ori.shape[1]/400)
    for cnt in list_contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Convert image to BGR (just for drawing a green rectangle on it).
        # bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(img_ori, [box], 0, (255, 0, 0), thickness)

    # Show images for debugging
    img_show = cv2.resize(img_ori, (800, 800))
    cv2.imshow('img show', img_show)
    cv2.waitKey(0)


        # x, y, w, h = cv2.boundingRect(cnt)
        # print('w', w * sw, ', h', h * sh)
        # if (w > size_min / 500 and h > size_min / 500):
        #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #     list_point = approx[:, 0]
        #     list_point = list_point.astype('float32')
        #     for p in list_point:
        #         p[0] *= sw
        #         p[1] *= sh
        #
        #     vertical = False
        #     if w * sw < h * sh: vertical = True
        #     quad = order_points(list_point, vertical=vertical)  # Sort lại rect theo thứ tự left top, right top...
        #     quad = quad.astype('int')
        #     list_quads.append(quad)
        #     if debug:
        #         for i in range(0, quad.shape[0] - 1):
        #             cv2.line(img_ori, (quad[i][0], quad[i][1]),
        #                      (quad[i + 1][0], quad[i + 1][1]), (0, 0, 255), 3)
        #         cv2.line(img_ori, (quad[0][0], quad[0][1]),
        #                  (quad[quad.shape[0] - 1][0], quad[quad.shape[0] - 1][1]), (0, 0, 255), 3)
        #         img_show = cv2.resize(img_ori, (800, 800))
        #         cv2.imshow('res', img_show)
        #         cv2.imshow('img ori', img_ori)
        #         # cv2.waitKey(0)
        #
        #     if len(list_point) != 4:
        #         print('find_quadrilateral. Failed!')
        #         return None

    list_quads.sort(key=lambda x: (x[0][0], x[0][1]))
    return list_quads


def transform_img(image, quad, size, dst):
    perspective_trans, status = cv2.findHomography(quad, dst)
    trans_img = cv2.warpPerspective(image, perspective_trans, (size[0], size[1]))
    return trans_img


# config_transform = {'idcard': [np.array([800, 504]),
#                                np.array([[21, 22], [779, 22], [779, 482], [21, 482]], dtype="float32")]}


def segment_and_rotate_img(input_img: np.ndarray, debug: bool) -> np.ndarray:
    '''
    segment vùng header trong ảnh phiếu ghi điểm golf và trả về ảnh đã được xoay lại cho thẳng
    :param input_img: opencv img
    :return: list của opencv img đã được căn chỉnh lại
    '''

    mask_img = onnx_model.inference(input_img)
    mask_img = mask_img[0][0].astype('uint8')

    mask_img[mask_img == 2] = 255
    mask_img = cv2.resize(mask_img, (input_img.shape[1], input_img.shape[0]))
    # mask_img[mask_img == 2] = 255

    # if debug:
    #     mask_img = cv2.resize(mask_img, (input_img.shape[1],input_img.shape[0]))
    #     cv2.imshow('mask', mask_img)
    #     cv2.waitKey(0)

    sw, sh = input_img.shape[1] / mask_img.shape[1], input_img.shape[0] / mask_img.shape[0]
    list_quads = find_quadrilateral(input_img, mask_img, sw=sw, sh=sh, debug=debug)
    #
    # list_calibed_imgs = []
    # if list_quads is None: return list_calibed_imgs
    #
    # for jdx, quad in enumerate(list_quads):
    #     draw_im = input_img.copy()
    #     for p in quad:
    #         cv2.circle(draw_im, (p[0], p[1]), 3, (0, 0, 255), -1)
    #     img_calibed = transform_img(input_img, quad, config_transform['idcard'][0], config_transform['idcard'][1])
    #     list_calibed_imgs.append(img_calibed)
    #
    #     if debug:
    #         cv2.imshow('draw img', draw_im)
    #         cv2.imshow('calib', img_calibed)
    #         cv2.waitKey(0)
    return input_img


def segment_src(input_src, output_dir, debug=False):
    '''

    :param input_src:
    :param output_dir:
    :param viz:
    :return:
    '''

    if os.path.isdir(input_src):
        list_files = get_list_file_in_folder(input_src)
        list_files = sorted(list_files)
        list_files = [os.path.join(input_src, f) for f in list_files]
        # color_map = visualize.get_color_map_list(256, custom_color=None)
    elif os.path.isfile(input_src):
        list_files = [input_src]
    begin = time.time()
    for idx, img_path in enumerate(list_files):
        if idx < 0: continue
        file = os.path.basename(img_path)
        print(idx, file)
        img_cv = cv2.imread(img_path)
        # list_calibed_imgs = segment_and_rotate_img(img_cv, debug=debug)

        # for jdx, img_calibed in enumerate(list_calibed_imgs):
        #     cv2.imwrite(os.path.join(output_dir, '.'.join(file.split('.')[:-1]) + '_' + str(jdx) + '.jpg'), img_calibed)
    end = time.time()
    print('avg inf time', 1000 * (end - begin) / len(list_files))

# from sanity_check.config.config_all import weight_dir
onnx_model = Segment_onnx(os.path.join('/home/misa/PycharmProjects/PaddleSeg/output/golf_header_model/iter_13k.onnx'),input_size=(960, 960))

if __name__ == '__main__':
    input_src = '/home/misa/PycharmProjects/MISA.ScoreCard/data/golf3/imgs'
    output_dir = '/home/misa/PycharmProjects/MISA.ScoreCard/data/res'
    segment_src(input_src=input_src,
                output_dir=output_dir,
                debug=True)
