from pycocotools.coco import COCO
import os, cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

'''
coco dataset should have structure likes
├── images
│   ├── 0755ecb7-golf2_028.jpeg
│   ├── 07fb2f8f-golf2_023.jpeg
│   ├── ...
└── result.json
'''

coco_dir = '/home/duycuong/PycharmProjects/PaddleSeg/data/golf_header_segmentation'
coco_anno_path = os.path.join(coco_dir, 'result.json')
save_anno_dir = '/home/duycuong/PycharmProjects/PaddleSeg/data/golf_header_segmentation/anno'
def coco2normal():
    coco = COCO(coco_anno_path)

    for idx in coco.imgs:
        img = coco.imgs[idx]
        img_path = os.path.join(coco_dir, img['file_name'])
        print(img_path)
        image = np.array(Image.open(img_path))
        plt.imshow(image, interpolation='nearest')
        # plt.show()
        # plt.imshow(image)
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        # coco.showAnns(anns)
        mask = np.zeros((img['height'],img['width']))
        for i in range(len(anns)):
            # kk= coco.annToMask(anns[i])
            mask += coco.annToMask(anns[i])*(1+anns[i]['category_id'])

        save_path = os.path.join(save_anno_dir,coco.imgs[idx]['file_name'].replace('.jpeg','.png'))
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        cv2.imwrite(save_path, mask)

    # anns_img = np.zeros((img['height'],img['width']))
    # for ann in anns:
    #     anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])

    # cv2.imshow('abc',mask)
    # cv2.waitKey(0)

    # cat_ids = coco.getCatIds()
    # anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    # anns = coco.loadAnns(anns_ids)
    # anns_img = np.zeros((img['height'],img['width']))
    # for ann in anns:
    #     anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])

if __name__ =='__main__':
    coco2normal()