# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os, cv2
import os.path as osp
import numpy as np
from PIL import Image, ImageDraw
from common import get_list_file_in_folder, resize_normalize
# from infer_onnx_find_quadrilateral import get_smallest_polygon_of_contour, transform_img, config_transform


class_name_to_id = {'_background_': 0,
                    'back12': 1,
                    'back9': 2,
                    'backchip': 3,
                    'front12': 4,
                    'front9': 5,
                    'frontchip': 6}

def shape2mask(img_size, points):
    label_mask = Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, 'Polygon must have points more than 2'
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)

def yolo2normal(input_imgs, output_imgs_dir, input_anno, output_anno_dir, resize_width = 2000):
    '''
    convert yolo annotation from LAbel studio to normal gray segmentation dataset
    :param input_imgs:
    :param output_imgs_dir:
    :param input_anno:
    :param output_anno_dir:
    :param resize_width:
    :return:
    '''
    if not osp.exists(output_anno_dir):
        os.makedirs(output_anno_dir)
    list_anno = get_list_file_in_folder(input_anno, ext=['txt'])
    for idx, label_file in enumerate(list_anno):
        if idx<0: continue
        print(idx, label_file)
        label_file = os.path.join(input_anno, label_file)
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            for ext in ['.jpg','.JPG','.png','.PNG','.jpeg','.JPEG']:
                img_file = osp.join(input_imgs, base +ext)
                if os.path.exists(img_file):
                    img = np.asarray(cv2.imread(img_file))
                    break

            label = np.zeros(img.shape[:2], dtype=np.int32)
            anno_lines = f.read()
            anno_lines = anno_lines.split('\n')
            h,w = img.shape[:2]
            for jdx, line in enumerate(anno_lines):
                if line == '': continue
                split_str = line.split(' ')

                num_pts = int((len(split_str) - 1) / 2)
                list_pts = []
                for num in range(num_pts):
                    x = float(split_str[1 + num * 2]) * w
                    y = float(split_str[1 + num * 2 + 1]) * h
                    pts = [x, y]
                    list_pts.append(pts)
                if len(list_pts)>2:
                    label_mask = shape2mask(img.shape[:2], list_pts)
                    label[label_mask] = int(split_str[0])+1
                    #cv2.imshow('mask', label)

            if resize_width is not None:
                img, ratio = resize_normalize(img, resize_width, interpolate=True)
                label, _ = resize_normalize(label, resize_width, interpolate=False)
                if ratio < 1: print('res ratio', ratio)
            out_png_file = osp.join(output_anno_dir, base + '.png')
            cv2.imwrite(out_png_file, label)
            cv2.imwrite(os.path.join(output_imgs_dir,base +'.jpg'), img)

def yolo2warpimg(input_anno, input_imgs, output_imgs_dir, debug = False):
    if not osp.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    list_anno = get_list_file_in_folder(input_anno, ext=['txt'])
    for f, label_file in enumerate(list_anno):
        # if '00402' not in label_file: continue
        print(f, label_file)
        img_basename = label_file.split('.')[0]
        label_file = os.path.join(input_anno, label_file)
        print('yolo2warpimg:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            img_file = osp.join(input_imgs, base +'.jpg')
            img = np.asarray(cv2.imread(img_file))
            anno_lines = f.read()
            anno_lines = anno_lines.split('\n')
            h,w = img.shape[:2]
            for n, line in enumerate(anno_lines):
                if line == '': continue
                split_str = line.split(' ')
                if split_str[0]!='1': continue

                num_pts = int((len(split_str) - 1) / 2)
                list_pts = []
                for num in range(num_pts):
                    x = float(split_str[1 + num * 2]) * w
                    y = float(split_str[1 + num * 2 + 1]) * h
                    pts = [x, y]
                    list_pts.append(pts)
                quad = np.asarray(list_pts)
                stt = True
                if len(quad) != 4:  # chi warp trong truong hop co 4 diem
                    quad = quad.reshape(quad.shape[0], 1, quad.shape[1])
                    quad = quad.astype('int')
                    stt, quad = get_smallest_polygon_of_contour(quad, 1.0, 1.0, min(w, h) / 5)
                    print('error' + 100 * '-')

                if stt:  # chi warp trong truong hop co 4 diem
                    img_calibed = transform_img(img, quad, config_transform['idcard'][0],
                                                config_transform['idcard'][1])
                    # total_warp += 1
                    if debug:
                        cv2.imshow('calib', img_calibed)
                        cv2.waitKey(0)

                    save_img_path = os.path.join(output_imgs_dir,
                                                 img_basename + '_' + str(n) + '.jpg')
                    cv2.imwrite(save_img_path, img_calibed)
                    # print('total_warp', total_warp, ', total_anno', total_anno)


if __name__ == '__main__':
    input_imgs = '/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/images'
    output_imgs_dir='/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/images_jpg'
    input_anno = '/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/labels'
    output_dir = '/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/labels_normal'
    yolo2normal(input_imgs=input_imgs,
                output_imgs_dir=output_imgs_dir,
                input_anno=input_anno,
                output_anno_dir=output_dir)


    # yolo2warpimg(input_anno=input_anno,
    #              input_imgs=input_imgs,
    #              output_imgs_dir=output_dir)