import os, cv2
import numpy as np
import random, shutil
import PIL.Image
import mmcv
import warnings


def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def get_list_file_in_dir_and_subdirs(folder, ext=['jpg', 'png', 'JPG', 'PNG']):
    file_names = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            extension = os.path.splitext(name)[1].replace('.', '')
            if extension in ext:
                file_names.append(os.path.join(path, name).replace(folder, '')[1:])
                # print(os.path.join(path, name).replace(folder,'')[1:])
    return file_names


def get_list_dir_and_subdirs_in_folder(folder):
    list_dir = [x[0].replace(folder, '').lstrip('/') for x in os.walk(folder)]
    return list_dir


def convert_anno_detection_to_segmentation(img_dir, anno_det_dir, output_anno_segment_dir, extend=-1,
                                           format_anno_det='icdar', class_list=dict()):
    list_images = get_list_file_in_folder(img_dir)
    list_images = sorted(list_images)
    for img_name in list_images:
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        anno_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        anno_file = os.path.join(anno_det_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        tree = open(anno_file, 'r', encoding='UTF-8')
        root = tree.readlines()
        for i, line in enumerate(root):
            line_str = line.split('\t')[0].replace('\n', '')
            idx = -1
            for i in range(0, 8):
                idx = line_str.find(',', idx + 1)

            coordinates = line_str[:idx]
            val = line_str[idx + 1:]
            left, top, right, _, _, bottom, _, _ = coordinates.split(",")
            cv2.rectangle(anno_mask, (int(left) - extend, int(top) - extend),
                          (int(right) + extend, int(bottom) + extend), 1, -1)
        cv2.imwrite(os.path.join(output_anno_segment_dir, img_name), anno_mask)


def split_dataset(img_dir, ann_dir, img_dst_dir, ann_dst_dir, ratio=0.5):
    list_images = get_list_file_in_folder(img_dir)
    random.shuffle(list_images)
    num_file = int(len(list_images) * ratio)
    print('split_dataset. Copy', num_file, 'files')
    for idx, img_name in enumerate(list_images):
        if idx > num_file:
            continue
        print(idx, img_name)
        ann_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(img_dst_dir, img_name))
        shutil.copy(os.path.join(ann_dir, ann_name), os.path.join(ann_dst_dir, ann_name))
    print('Done')


def del_dataset(img_dir, ann_dir):
    list_images = get_list_file_in_folder(img_dir)
    list_images = sorted(list_images)
    for idx, img_name in enumerate(list_images):
        print(idx, img_name)
        ann_path = os.path.join(ann_dir, img_name.replace('.jpg', '.png'))
        if not os.path.exists(ann_path):
            os.remove(os.path.join(img_dir, img_name))
    print('Done')


def refine_dataset(img_dir, ann_dir):
    list_images = get_list_file_in_folder(img_dir)
    parent_dir = os.path.dirname(img_dir)
    output_dir = os.path.join(parent_dir, 'img_wo_anno')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, file in enumerate(list_images):
        img_path = os.path.join(img_dir, file)
        anno_file = os.path.join(ann_dir, file.replace('.jpg', '.png'))
        if not os.path.exists(anno_file):
            print(idx, file)
            print('----------------------------------------------------------------------------------')
            shutil.move(img_path, os.path.join(output_dir, file))
            kk = 1


def refactor_classes_of_dataset(src_anno_dir, dst_anno_dir,
                                src_classes=[1, 2, 3, 4, 5],
                                dst_classes=[1, 1, 3, 2, 1]):
    src_dict = {}
    for i in range(len(src_classes)):
        src_dict[i + 1] = dst_classes[i]

    from numpy import copy

    list_images = get_list_file_in_folder(src_anno_dir)
    list_images = sorted(list_images)
    for idx, file in enumerate(list_images):
        print(idx, file)
        src_img_path = os.path.join(src_anno_dir, file)
        dst_img_path = os.path.join(dst_anno_dir, file)
        img_data = cv2.imread(src_img_path, 0)
        newArray = copy(img_data)
        # for k, v in src_dict.items():
        #     newArray[img_data == k] = v
        cv2.imwrite(dst_img_path, newArray)


def convert_all_imgs_to_jpg(src_anno_dir, dst_anno_dir):
    ext = ['jpg', 'png', 'JPG', 'PNG']
    list_images = get_list_file_in_folder(src_anno_dir, ext=ext)
    list_images = sorted(list_images)
    for idx, file in enumerate(list_images):
        print(idx, file)
        src_img_path = os.path.join(src_anno_dir, file)
        for e in ext:
            file = file.replace(e, 'jpg')
        dst_img_path = os.path.join(dst_anno_dir, file)
        img_data = cv2.imread(src_img_path)
        cv2.imwrite(dst_img_path, img_data)


def convert_voc_label_to_normal_format(src_anno_dir, dst_anno_dir):
    print('convert_voc_label_to_normal_format')
    print('src_anno_dir', src_anno_dir)
    print('dst_anno_dir', dst_anno_dir)

    list_images = get_list_file_in_folder(src_anno_dir)
    list_images = sorted(list_images)
    for idx, file in enumerate(list_images):
        print(idx, file)
        src_img_path = os.path.join(src_anno_dir, file)
        dst_img_path = os.path.join(dst_anno_dir, file)
        lbl = np.asarray(PIL.Image.open(src_img_path))
        cv2.imwrite(dst_img_path, lbl)


def show_result(img,
                result,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]
    # if palette is None:
    #     if self.PALETTE is None:
    #         palette = np.random.randint(
    #             0, 255, size=(len(self.CLASSES), 3))
    #     else:
    #         palette = self.PALETTE
    palette = np.array(palette)
    # assert palette.shape[0] == len(self.CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    if not (show or out_file):
        warnings.warn('show==False and out_file is not specified, only '
                      'result image will be returned')
        return img


def visualize_normal_format_dataset(img_dir, ann_dir, viz_dir=None):
    PALETTE = [
        [128, 64, 128],
        [120, 120, 120],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [244, 35, 232],
        [70, 140, 70],
        [0, 0, 180],
        [0, 0, 180],
        [190, 0, 153],
        [100, 180, 120],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [255, 64, 64],
        [200, 35, 100],
        [140, 0, 70],
        [200, 102, 102],
        [153, 0, 153],
        [30, 153, 20],
        [150, 50, 50],
        [150, 30, 250],
        [60, 200, 60]
    ]
    ext = ['jpg', 'jpeg', 'png', 'JPG', 'PNG']
    list_images = get_list_file_in_folder(img_dir, ext=ext)
    list_images = sorted(list_images)
    for idx, file in enumerate(list_images):
        print(idx, file)
        # if idx <200:
        #     continue
        img_path = os.path.join(img_dir, file)
        for e in ext:
            file = file.replace('.' + e, '.png')
        ann_path = os.path.join(ann_dir, file)
        if not os.path.exists(ann_path):
            print('Anno file not exist!')
            continue

        img_data = cv2.imread(img_path)
        ann_data = np.asarray(PIL.Image.open(ann_path))
        out_file = None
        if viz_dir is not None:
            out_file = os.path.join(viz_dir, file)
        show_result(
            img_data,
            [ann_data],
            palette=PALETTE,
            show=True,
            out_file=out_file)


if __name__ == '__main__':
    # split_dataset(img_dir='/data4T/cuongnd/dataset/publaynet_split1/img_dir/train',
    #               ann_dir='/data4T/cuongnd/dataset/publaynet_split1/ann_dir/train_3classes',
    #               img_dst_dir='/data4T/cuongnd/dataset/doc_structure1/img_dir/train',
    #               ann_dst_dir='/data4T/cuongnd/dataset/doc_structure1/ann_dir/train',
    #               ratio=0.002)

    # src_anno_dir='/data_backup/cuongnd/mmseg/doc_seg/imgs/test'
    # dst_anno_dir='/data_backup/cuongnd/mmseg/doc_seg/imgs/test'
    # convert_all_imgs_to_jpg(src_anno_dir,dst_anno_dir)

    img_dir = '/home/duycuong/PycharmProjects/PaddleSeg/data/golf_header_segmentation/images'
    ann_dir = '/home/duycuong/PycharmProjects/PaddleSeg/data/golf_header_segmentation/anno/images'
    viz_dir = '/home/duycuong/PycharmProjects/PaddleSeg/data/golf_header_segmentation/viz'
    visualize_normal_format_dataset(img_dir=img_dir,
                                    ann_dir=ann_dir,
                                    viz_dir=viz_dir)
