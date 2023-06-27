from pycocotools.coco import COCO
import os, cv2
from PIL import Image
import numpy as np
from infer_onnx_find_quadrilateral import get_smallest_polygon_of_contour, transform_img, config_transform
from common import resize_normalize

'''
coco dataset should have structure likes
├── images
│   ├── 0755ecb7-golf2_028.jpg
│   ├── 07fb2f8f-golf2_023.jpg
│   ├── ...
└── result.json
'''

coco_dir = '/home/misa/Downloads/doc_seg_coco'
coco_anno_path = os.path.join(coco_dir, 'result.json')

save_img_dir = '/home/misa/Downloads/testset1k_doc_rot/ims'
save_anno_dir = '/home/misa/Downloads/testset1k_doc_rot/labels'

warp_dir = '/home/misa/Downloads/doc_seg_coco/warp_imgs'


def coco2warpimg(debug=False):
    '''
    warp dữ liệu cmnd theo format coco
    :return:
    '''
    coco = COCO(coco_anno_path)
    total_anno = 0
    total_warp = 0
    for idx in coco.imgs:
        img = coco.imgs[idx]
        img_path = os.path.join(coco_dir, img['file_name'])
        img_extention = '.' + img['file_name'].split('.')[-1]
        img_basename = os.path.basename(img['file_name']).split('-')[0]
        print(idx, img_path)
        image = np.array(Image.open(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        w, h = image.shape[1], image.shape[0]

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        for n, ann in enumerate(anns):
            total_anno += 1
            quad = []
            for i in range(int(len(ann['segmentation'][0]) / 2)):
                x = ann['segmentation'][0][2 * i]
                y = ann['segmentation'][0][2 * i + 1]
                quad.append([x, y])
            quad = np.asarray(quad)
            stt = True
            if len(quad) != 4:  # chi warp trong truong hop co 4 diem
                quad = quad.reshape(quad.shape[0], 1, quad.shape[1])
                quad = quad.astype('int')
                stt, quad = get_smallest_polygon_of_contour(quad, 1.0, 1.0, min(w, h) / 5)
                print('error' + 100 * '-')

            if stt:  # chi warp trong truong hop co 4 diem
                img_calibed = transform_img(image, quad, config_transform['idcard'][0],
                                            config_transform['idcard'][1])
                total_warp += 1
                if debug:
                    cv2.imshow('calib', img_calibed)
                    cv2.waitKey(0)

                save_img_path = os.path.join(warp_dir, img_basename.replace(img_extention, '') + '_' + str(n) + '.jpg')
                cv2.imwrite(save_img_path, img_calibed)
                print('total_warp', total_warp, ', total_anno', total_anno)



def coco2normal(resize_width=1200):
    '''

    :param fix_extension: True: convert tất cả sang jpg
    :param resize_max: Kích cỡ tối đa resize cạnh dài về
    :return:
    '''
    if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.exists(save_anno_dir): os.makedirs(save_anno_dir)
    coco = COCO(coco_anno_path)
    for idx in coco.imgs:
        img = coco.imgs[idx]
        img_path = os.path.join(coco_dir, img['file_name'])
        img_extention = '.' + img['file_name'].split('.')[-1]
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
        mask = np.zeros((img['height'], img['width']))
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i]) * (1 + anns[i]['category_id'])

        save_anno_path = os.path.join(save_anno_dir, img_basename.replace(img_extention, '.png'))
        save_img_path = os.path.join(save_img_dir, img_basename.replace(img_extention, '.jpg'))

        if resize_width is not None:
            image, ratio = resize_normalize(image, 1200, interpolate=True)
            mask, _ = resize_normalize(mask, 1200, interpolate=False)
            if ratio<1: print('res ratio', ratio)
        cv2.imwrite(save_anno_path, mask)
        cv2.imwrite(save_img_path, image)


def test():
    img = cv2.imread(
        '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg/labels/0b5f9110-5b1e13e7951c443499af8979494de881638190468996862934.png')
    res_img = cv2.resize(img, (2000, 2000), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('sample.png', res_img)


if __name__ == '__main__':
    coco2warpimg()
    # test()
