from pycocotools.coco import COCO
import os, cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from common import resize_normalize

'''
coco dataset should have structure likes
├── images
│   ├── 0755ecb7-golf2_028.jpg
│   ├── 07fb2f8f-golf2_023.jpg
│   ├── ...
└── result.json
'''

coco_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg3'
coco_anno_path = os.path.join(coco_dir, 'result.json')

save_img_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg3/images_jpg'
if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
save_anno_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg3/labels'
if not os.path.exists(save_anno_dir): os.makedirs(save_anno_dir)


def coco2normal(resize_width = 1200):
    '''

    :param fix_extension: True: convert tất cả sang jpg
    :param resize_max: Kích cỡ tối đa resize cạnh dài về
    :return:
    '''
    coco = COCO(coco_anno_path)
    for idx in coco.imgs:
        img = coco.imgs[idx]
        img_path = os.path.join(coco_dir, img['file_name'])
        img_extention = '.'+ img['file_name'].split('.')[-1]
        img_basename = os.path.basename(img['file_name'])
        print(idx, img_path)
        image = np.array(Image.open(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # plt.imshow(image, interpolation='nearest')
        # plt.show()
        # plt.imshow(image)
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = np.zeros((img['height'],img['width']))
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])*(1+anns[i]['category_id'])

        save_anno_path = os.path.join(save_anno_dir,img_basename.replace(img_extention,'.png'))
        save_img_path = os.path.join(save_img_dir,img_basename.replace(img_extention,'.jpg'))

        if resize_width is not None:
            image, ratio = resize_normalize(image, 1200, interpolate=True)
            mask, _ = resize_normalize(mask, 1200, interpolate=False)
            if ratio<1: print('res ratio', ratio)
        cv2.imwrite(save_anno_path, mask)
        cv2.imwrite(save_img_path, image)


def test():
    img = cv2.imread('/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg/labels/0b5f9110-5b1e13e7951c443499af8979494de881638190468996862934.png')
    res_img = cv2.resize(img,(2000,2000),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite('sample.png', res_img)


if __name__ =='__main__':
    coco2normal()
    #test()