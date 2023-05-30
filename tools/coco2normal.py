from pycocotools.coco import COCO
import os, cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

'''
coco dataset should have structure likes
├── images
│   ├── 0755ecb7-golf2_028.jpg
│   ├── 07fb2f8f-golf2_023.jpg
│   ├── ...
└── result.json
'''

coco_dir = '/home/misa/PycharmProjects/MISA.ScoreCard/data/golf_header'
coco_anno_path = os.path.join(coco_dir, 'result.json')
save_anno_dir = '/home/misa/PycharmProjects/MISA.ScoreCard/data/golf_header/labels'
if not os.path.exists(save_anno_dir): os.makedirs(save_anno_dir)

def coco2normal():
    coco = COCO(coco_anno_path)

    for idx in coco.imgs:
        img = coco.imgs[idx]
        img_path = os.path.join(coco_dir, img['file_name'])
        img_extention = '.'+ img['file_name'].split('.')[-1]
        img_basename = os.path.basename(img['file_name'])
        print(idx, img_path)
        image = np.array(Image.open(img_path))
        plt.imshow(image, interpolation='nearest')
        # plt.show()
        # plt.imshow(image)
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        mask = np.zeros((img['height'],img['width']))
        for i in range(len(anns)):
            # kk= coco.annToMask(anns[i])
            mask += coco.annToMask(anns[i])*(1+anns[i]['category_id'])

        save_path = os.path.join(save_anno_dir,img_basename.replace(img_extention,'.png'))
        cv2.imwrite(save_path, mask)


if __name__ =='__main__':
    coco2normal()