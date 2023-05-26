from pycocotools.coco import COCO
import os, cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

coco = COCO('/home/misa/PycharmProjects/MISA.ScoreCard/data/golf_header_segmentation/result.json')
img_dir = '/home/misa/PycharmProjects/MISA.ScoreCard/data/golf_header_segmentation'
image_id = 10

img = coco.imgs[image_id]
# loading annotations into memory...
# Done (t=12.70s)
# creating index...
# index created!
image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
plt.imshow(image, interpolation='nearest')
# plt.show()
# plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
# coco.showAnns(anns)
mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    kk= coco.annToMask(anns[i])
    mask += coco.annToMask(anns[i])

# anns_img = np.zeros((img['height'],img['width']))
# for ann in anns:
#     anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])

cv2.imshow('abc',mask)
cv2.waitKey(0)

# cat_ids = coco.getCatIds()
# anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
# anns = coco.loadAnns(anns_ids)
# anns_img = np.zeros((img['height'],img['width']))
# for ann in anns:
#     anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])
