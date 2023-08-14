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

import os, cv2, random
import numpy as np
from common import get_list_file_in_folder, resize_normalize
from yolo2normal import shape2mask


def yoloseg2det(anno_dir,
                save_crop_image_and_crop_anno=True,
                imgs_dir=None,
                resize_width=800):
    '''
    1. convert yolo segmentation annotation to yolo detection annotation
    2. Lưu dataset được crop phục vụ cho training segmentation
    :param anno_dir: thư mục chứa yolo segmentation annotation
    :param save_crop_image_and_crop_anno: Có lưu dataset segmentation được crop hay ko --> phục vụ training segmentation sau khi được crop từ detection
    :param imgs_dir: thư mục chứa ảnh
    :param resize_width:
    :return:
    '''

    list_anno = get_list_file_in_folder(anno_dir, ext=['txt'])
    # for detection
    output_yolo_det_anno = anno_dir +'_yolo_det'
    os.makedirs(output_yolo_det_anno, exist_ok=True)
    output_yolo_det_img = imgs_dir +'_res'
    os.makedirs(output_yolo_det_img, exist_ok=True)

    #for segmentation
    if save_crop_image_and_crop_anno:
        output_anno_dir = imgs_dir + '_anno_crop'
        if not os.path.exists(output_anno_dir): os.makedirs(output_anno_dir)
        output_imgs_dir = imgs_dir + '_imgs_crop'
        if not os.path.exists(output_imgs_dir): os.makedirs(output_imgs_dir)
    print('Generating dataset from:', anno_dir)
    total_count=0
    for idx, label_file in enumerate(list_anno):
        if idx < 0: continue
        # if 'f1c1116f-cccd_ffd87b8719674dc29523d8efeb63a481637995166662272115' not in label_file:
        #     continue
        print(idx, label_file)

        label_file = os.path.join(anno_dir, label_file)
        with open(label_file) as f:
            final_line = []
            base = os.path.splitext(os.path.basename(label_file))[0]
            if imgs_dir is not None:
                for ext in ['jpg','JPG','png','PNG']:
                    if os.path.exists(os.path.join(imgs_dir, base + '.'+ext)):
                        img_file = os.path.join(imgs_dir, base + '.'+ext)
                img = np.asarray(cv2.imread(img_file))
                img_w, img_h = img.shape[1], img.shape[0]

            anno_lines = f.read()
            anno_lines = anno_lines.split('\n')
            count =0
            for jdx, line in enumerate(anno_lines):
                if save_crop_image_and_crop_anno and imgs_dir is not None:
                    label = np.zeros(img.shape[:2], dtype=np.int32)
                if line == '': continue
                count +=1
                split_str = line.split(' ')
                num_pts = int((len(split_str) - 1) / 2)

                #2. save crop image and labels for segmentation
                min_x, min_y, max_x, max_y = 99999, 99999, -1, -1
                list_pts = []
                for num in range(num_pts):
                    x = float(split_str[1 + num * 2])*img_w
                    y = float(split_str[1 + num * 2 + 1]) *img_h
                    pts = [x, y]
                    list_pts.append(pts)
                    if x > max_x: max_x = x
                    if y > max_y: max_y = y
                    if x < min_x: min_x = x
                    if y < min_y: min_y = y

                if len(list_pts) > 2 and save_crop_image_and_crop_anno and imgs_dir is not None:
                    label_mask = shape2mask(img.shape[:2], list_pts)
                    label[label_mask] = 1
                    extend = random.uniform(0.0, 0.1)
                    w, h = max_x - min_x, max_y - min_y
                    extend_x, extend_y = extend * w, extend * h
                    crop_x1 = int(max(0, min_x - extend_x))
                    crop_y1 = int(max(0, min_y - extend_y))
                    crop_x2 = int(min(img_w - 1, max_x + extend_x))
                    crop_y2 = int(min(img_h - 1, max_y + extend_y))
                    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    crop_label = label[crop_y1:crop_y2, crop_x1:crop_x2]
                    if resize_width is not None:
                        crop_img, ratio = resize_normalize(crop_img, resize_width, interpolate=True)
                        crop_label, _ = resize_normalize(crop_label, resize_width, interpolate=False)
                    cv2.imwrite(os.path.join(output_imgs_dir, base + '_{}.jpg'.format(str(jdx))), crop_img)
                    cv2.imwrite(os.path.join(output_anno_dir, base + '_{}.png'.format(str(jdx))), crop_label)

                #1. save anno yolo detection + resize image for better disk usage
                x_center = np.around((max_x+min_x)/(2*img_w), decimals=2)
                y_center = np.around((max_y+min_y)/(2*img_h), decimals=2)
                w_relative = np.around((max_x-min_x)/img_w, decimals=2)
                h_relative = np.around((max_y-min_y)/img_h, decimals=2)
                save_line = ' '.join([split_str[0],str(x_center), str(y_center), str(w_relative), str(h_relative)])
                final_line.append(save_line)

            resize_img, _ = resize_normalize(img, 1000, interpolate=True)
            cv2.imwrite(os.path.join(output_yolo_det_img, base + '.jpg'), resize_img)

            save_yolo_anno = '\n'.join(final_line)
            with open(os.path.join(output_yolo_det_anno, base+'.txt'), mode='w') as fi:
                fi.write(save_yolo_anno)

            # if count>5:
            #     print('image has more than 5 objects', count)
            total_count +=count
    print('total anno', total_count)


if __name__ == '__main__':
    imgs_dir = '/home/misa/Downloads/project-55-at-2023-08-07-03-58-2bfd95e6/images'
    anno_dir = '/home/misa/Downloads/project-55-at-2023-08-07-03-58-2bfd95e6/labels_seg'
    yoloseg2det(anno_dir=anno_dir,
                save_crop_image_and_crop_anno=True,
                imgs_dir=imgs_dir,
                resize_width=800)
