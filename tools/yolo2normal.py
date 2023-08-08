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
import os, cv2, math, shutil
import os.path as osp
import numpy as np
from PIL import Image, ImageDraw
from common import get_list_file_in_folder, resize_normalize


class_name_to_id = {'_background_': 0,
                    'back12': 1,
                    'back9': 2,
                    'backchip': 3,
                    'front12': 4,
                    'front9': 5,
                    'frontchip': 6,
                    'passport':7}
id_to_class_name = {}
for key in class_name_to_id:
    id_to_class_name[class_name_to_id[key]] = key

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

config_transform = {'idcard': [np.array([800, 504]),
                               np.array([[21, 22], [779, 22], [779, 482], [21, 482]], dtype="float32")]}

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

def transform_img(image, quad, size, dst):
    perspective_trans, status = cv2.findHomography(quad, dst)
    trans_img = cv2.warpPerspective(image, perspective_trans, (size[0], size[1]))
    return trans_img

def order_points(pts):
    '''
    Sắp xếp lại các điểm phục vụ cho transformation
    :param pts:
    :return:
    '''
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    if euclidean_distance(rect[0],rect[1])<euclidean_distance(rect[1],rect[2]):
        new_rec = [rect[1],rect[2],rect[3],rect[0]]
        rect = np.asarray(new_rec)
    return rect

def find_bounding_poly(contour, min_size):
    '''
    tìm tứ giác hoặc hình chữ nhật nhỏ nhất (trong trường hợp kết quả segment ko đủ tốt) bao quanh contour
    :param contour:
    :param min_size:
    :return:
    '''
    x, y, w, h = cv2.boundingRect(contour)
    if (w > min_size and h > min_size):
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        list_point = approx[:, 0]
        list_point = list_point.astype('float32')
        if len(list_point)!=4:
            rect = cv2.minAreaRect(contour)
            list_point = np.intp(cv2.boxPoints(rect))
        return True, list_point.astype('int').tolist(), [x,y,w,h]
    print('find_bounding_poly. Failed!')
    return False, None, [x,y,w,h]

def yolo2warpimg(input_anno, input_imgs, output_imgs_dir, debug = False):
    '''

    :param input_anno:
    :param input_imgs:
    :param output_imgs_dir:
    :param debug:
    :return:
    '''
    if not osp.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    list_anno = get_list_file_in_folder(input_anno, ext=['txt'])
    count = 0
    for f, label_file in enumerate(list_anno):
        # if '0f2d7a04-cccd_e0263874d4524c16b7090a3cbef38a91638194458626955393' not in label_file: continue
        if f < 0: continue
        print(f, label_file)
        img_basename = label_file.split('.')[0]
        label_file = os.path.join(input_anno, label_file)
        print('yolo2warpimg:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            for ext in ['jpg','JPG','png','PNG']:
                img_file = osp.join(input_imgs, base +'.'+ext)
                if os.path.exists(img_file):
                    break
            img = np.asarray(cv2.imread(img_file))
            anno_lines = f.read()
            anno_lines = anno_lines.split('\n')
            h,w = img.shape[:2]
            for n, line in enumerate(anno_lines):
                if line == '': continue
                split_str = line.split(' ')
                # if split_str[0]!='1': continue
                count+=1
                cls = id_to_class_name[int(split_str[0])+1]

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
                    stt, quad, bbox = find_bounding_poly(quad, min(w, h) / 10)

                quad = order_points(np.asarray(quad))  # Sort lại rect theo thứ tự left top, right top...
                if len(quad) != 4:
                    print('error' + 100 * '-')
                if stt:  # chi warp trong truong hop co 4 diem
                    img_calibed = transform_img(img, np.asarray(quad), config_transform['idcard'][0],
                                                config_transform['idcard'][1])
                    # total_warp += 1
                    if debug:
                        cv2.imshow('calib', img_calibed)
                        cv2.waitKey(0)
                    out_dir = os.path.join(output_imgs_dir, cls)
                    if not os.path.exists(out_dir): os.makedirs(out_dir)
                    save_img_path = os.path.join(out_dir,
                                                 img_basename + '_' + str(n) + '.jpg')
                    cv2.imwrite(save_img_path, img_calibed)
                    # print('total_warp', total_warp, ', total_anno', total_anno)
    print('Total objects in annotation', count)

def yolo2nondewarpimg(input_anno, input_imgs, output_imgs_dir, debug = False):
    '''

    :param input_anno:
    :param input_imgs:
    :param output_imgs_dir:
    :param debug:
    :return:
    '''
    if not osp.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
    list_anno = get_list_file_in_folder(input_anno, ext=['txt'])
    count = 0
    all_2_lines = []
    for f, label_file in enumerate(list_anno):
        # if '0f2d7a04-cccd_e0263874d4524c16b7090a3cbef38a91638194458626955393' not in label_file: continue
        print(f, label_file)
        img_basename = label_file.split('.')[0]
        label_file = os.path.join(input_anno, label_file)
        print('yolo2nondewarpimg:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            img_file = osp.join(input_imgs, base +'.jpg')
            img = np.asarray(cv2.imread(img_file))
            anno_lines = f.read()
            anno_lines = anno_lines.split('\n')
            h,w = img.shape[:2]
            new_anno_lines = []
            for n, line in enumerate(anno_lines):
                if line == '': continue
                new_anno_lines.append(line)
            anno_lines = new_anno_lines

            if len(anno_lines)==2:
                all_2_lines.append(label_file)
            elif len(anno_lines)==1:
                save_img_path = os.path.join(output_imgs_dir,
                                             img_basename + '_0.jpg')
                shutil.move(img_file, save_img_path)
                    # print('total_warp', total_warp, ', total_anno', total_anno)
    print('all 2 files')
    for idx, line in enumerate(all_2_lines):
        print(idx, line)

if __name__ == '__main__':
    input_imgs = '/home/misa/PycharmProjects/MISA.eKYC2/data/evaluation/ekyc_doc_seg/v2/images'
    # output_imgs_dir='/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/images_jpg'
    input_anno = '/home/misa/PycharmProjects/MISA.eKYC2/data/evaluation/ekyc_doc_seg/v2/labels_seg'
    output_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/evaluation/ekyc_doc_seg/v2/dewarp'
    # yolo2normal(input_imgs=input_imgs,
    #             output_imgs_dir=output_imgs_dir,
    #             input_anno=input_anno,
    #             output_anno_dir=output_dir)


    yolo2warpimg(input_anno=input_anno,
                 input_imgs=input_imgs,
                 output_imgs_dir=output_dir)