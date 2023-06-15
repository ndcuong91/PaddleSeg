#!/usr/bin/env python
# coding: utf-8
import os
import matplotlib.pyplot as plt
import paddleseg.transforms as T
from paddleseg.datasets import Dataset


transforms = [T.ResizeStepScaling(0.125, 1.5, 0.125), 
              T.RandomPaddingCrop((1024, 512)), 
              T.RandomHorizontalFlip(),
              T.RandomDistort(0.5, 0.5, 0.5), 
              T.Normalize()]
dataset_root = '/home/misa/PycharmProjects/PaddleSeg/data/ekyc_doc_seg1234/testset1000/images'
train_path = '/home/misa/PycharmProjects/PaddleSeg/data/ekyc_doc_seg1234/test.txt'
num_classes = 2
dataset = Dataset(transforms = transforms,
                    dataset_root = dataset_root,
                    num_classes = 2,
                    train_path = train_path,
                    mode = 'train')


augment_img = dataset[0]['img']
plt.imshow(augment_img.transpose(1, 2, 0))
img_names = os.listdir(dataset_root)

print(len(img_names))
print(img_names[:5])


# In[24]:


n = 12
num_rows, num_cols = 2, 6 
fig = plt.figure(figsize=(20, 5))

for i in range(n):
    if i < 6: 
        img = plt.imread(os.path.join(dataset_root, img_names[i]))

        fig.add_subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    else:
        trans_img = dataset[i - 6]['img'].transpose(1, 2, 0)

        mean = (0.5, )
        std = (0.5, )
        trans_img = std * trans_img + mean

        fig.add_subplot(num_rows, num_cols, i + 1)
        plt.imshow(trans_img)
        plt.axis('off')

plt.show()
