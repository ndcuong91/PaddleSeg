import os, cv2
from common import get_list_file_in_folder, get_list_file_in_dir_and_subdirs
import random
def create_doc_rot_test(input_dir, output_dir):
    '''

    :param input_dir:
    :return:
    '''

    list_files = get_list_file_in_folder(input_dir)
    list_line = []
    for idx, f in enumerate(list_files):
        print(idx, f)
        img_path = os.path.join(input_dir, f)
        output_path = os.path.join(output_dir, f)

        rotate = random.choice([True, False])
        img = cv2.imread(img_path)
        line = ' '.join([f,str(0)])
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
            line = ' '.join([f,str(1)])
        list_line.append(line)
        cv2.imwrite(output_path, img)
    list_line = '\n'.join(list_line)
    with open(os.path.join(output_dir, 'gt.txt'), mode = 'w') as f:
        f.write(list_line)

import random, shutil

def split_train_val(input_img, input_anno, train_ratio = 0.8):
    '''

    :param input_dir:
    :return:
    '''

    list_files = get_list_file_in_folder(input_img)
    list_line = []

    num_train = int(train_ratio*len(list_files))
    random.shuffle(list_files)
    list_train = list_files[:num_train]
    list_val = list_files[num_train:]

    train_img_dir = os.path.join(input_img,'train')
    if not os.path.exists(train_img_dir): os.makedirs(train_img_dir)
    val_img_dir = os.path.join(input_img,'val')
    if not os.path.exists(val_img_dir): os.makedirs(val_img_dir)
    train_anno_dir = os.path.join(input_anno,'train')
    if not os.path.exists(train_anno_dir): os.makedirs(train_anno_dir)
    val_anno_dir = os.path.join(input_anno,'val')
    if not os.path.exists(val_anno_dir): os.makedirs(val_anno_dir)

    for idx, file in enumerate(list_train):
        img_path = os.path.join(input_img, file)
        base_name = file.split('.')[0]
        shutil.move(img_path, os.path.join(train_img_dir, file))
        shutil.move(os.path.join(input_anno, base_name+'.txt'),os.path.join(train_anno_dir, base_name+'.txt'))

    for idx, file in enumerate(list_val):
        img_path = os.path.join(input_img, file)
        base_name = file.split('.')[0]
        shutil.move(img_path, os.path.join(val_img_dir, file))
        shutil.move(os.path.join(input_anno, base_name+'.txt'),os.path.join(val_anno_dir, base_name+'.txt'))


def convert_rgbgray(anno_dir):
    '''

    '''
    output_anno_dir = anno_dir+'_gray'
    if not os.path.exists(output_anno_dir): os.makedirs(output_anno_dir)
    list_anno = get_list_file_in_folder(anno_dir)

    for idx, anno in enumerate(list_anno):
        print(idx, anno)
        annoimg = cv2.imread(os.path.join(anno_dir, anno), 0)
        # cv2.imshow('abc', annoimg)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(output_anno_dir, anno), annoimg)

def resize_all_data_in_dir(input_dir, res = (600,400)):
    output_dir = input_dir+'_resize'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    list_files = get_list_file_in_dir_and_subdirs(input_dir)

    for idx, file in enumerate(list_files):
        print(idx, file)
        file_path = os.path.join(input_dir, file)
        dst_file_path = os.path.join(output_dir, file)
        dir_name = os.path.dirname(dst_file_path)
        if not os.path.exists(dir_name): os.makedirs(dir_name)
        img = cv2.imread(file_path)
        img = cv2.resize(img, res)
        cv2.imwrite(dst_file_path, img)


if __name__=='__main__':
    # input_dir = '/home/haongnd/id_classification/data_resize/to_detection'
    # output_dir = '/home/haongnd/id_classification/data_resize/to_detection_test'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # create_doc_rot_test(input_dir, output_dir)

    split_train_val(input_img='/home/misa/Downloads/project-230-at-2023-12-27-04-53-dbfcb244/ekyc_doc_det_v2/images',
                    input_anno='/home/misa/Downloads/project-230-at-2023-12-27-04-53-dbfcb244/ekyc_doc_det_v2/labels')

    # convert_rgbgray('/home/misa/Downloads/project-55-at-2023-06-26-09-48-2a8e1279/labels_crop2')

    # resize_all_data_in_dir('/home/haongnd/id_classification/original_data')

