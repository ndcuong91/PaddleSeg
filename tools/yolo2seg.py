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

import glob
import os
import os.path as osp
import numpy as np
from PIL import Image
import PIL.ImageDraw
import cv2
from common import get_list_file_in_folder

input_imgs = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg4/images'
input_anno = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg4/labels'
output_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg4/anno'
if not osp.exists(output_dir):
    os.makedirs(output_dir)
    print('Creating annotations directory:', output_dir)
class_name_to_id = {'_background_': 0,
                    'back12': 1,
                    'back9': 2,
                    'backchip': 3,
                    'front12': 4,
                    'front9': 5,
                    'frontchip': 6}

def main():
    list_anno = get_list_file_in_folder(input_anno, ext=['txt'])
    for f, label_file in enumerate(list_anno):
        print(f, label_file)
        label_file = os.path.join(input_anno, label_file)
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            img_file = osp.join(input_imgs, base +'.jpg')
            img = np.asarray(cv2.imread(img_file))

            #img = np.array(Image.open(img_file))
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            label = np.zeros(img.shape[:2], dtype=np.int32)
            anno_lines = f.read()
            anno_lines = anno_lines.split('\n')
            h,w = img.shape[:2]
            for line in anno_lines:
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

            out_png_file = osp.join(output_dir, base + '.png')
            cv2.imwrite(out_png_file, label)


def shape2mask(img_size, points):
    label_mask = PIL.Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = PIL.ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, 'Polygon must have points more than 2'
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)


if __name__ == '__main__':
    # args = parse_args()
    # args.input_dir = input_dir
    main()
