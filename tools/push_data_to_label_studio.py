from label_studio_sdk import Client
import os, cv2
from PIL import Image

LABEL_STUDIO_URL = 'http://10.8.28.25:9001'
API_KEY = '14086a1174df7d7be496b8ac6e4e5fa6cd87380f'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)


def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    file_names = sorted(file_names)
    return file_names


cls_map = {
    0: 'front9',
    1: 'back9',
    2: 'front12',
    3: 'back12',
    4: 'frontchip',
    5: 'backchip'
}


def push_yolo_dataset_to_label_studio(proj_id, img_dir, anno_dir, server_img_dir, cls_file=None, export_version=None):
    '''
    1. copy ảnh từ thư mục img_dir vào thư mục server_img_dir trên server
    2. Chạy script tạo project mới
    3. sửa config "add cloud storage" và link đến thư mục server_img_dir

    :param proj_name:
    :param img_dir:
    :param anno_dir:
    :param server_img_dir: Thư mục chứa img trên docker của label-studio
    :param cls_file:
    :return:
    '''
    project = ls.get_project(proj_id)
    project.set_params(label_config='''
      <View>
      <Header value="Select label and click the image to start"/>
      <Image name="image" value="$image" zoom="true"/>
      <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small" opacity="0.5">
      <Label value="front9" background="#FFA39E"/><Label value="back9" background="#0dd349"/><Label value="front12" background="#FFC069"/><Label value="back12" background="#001dad"/><Label value="frontchip" background="#f2ad5f"/><Label value="backchip" background="#8461b8"/></PolygonLabels>
    </View>
        ''')

    list_imgs = get_list_file_in_folder(img_dir)

    for idx, img_name in enumerate(list_imgs):
        #if idx > 10: continue
        print(idx, img_name)
        task = {'data': {'image': os.path.join(server_img_dir, img_name)},
                'predictions': [{'result': []}]}
        img = cv2.imread(os.path.join(img_dir, img_name))
        w, h = img.shape[1], img.shape[0]
        anno_path = os.path.join(anno_dir, img_name.replace('.jpg', '.txt'))
        if not os.path.exists(anno_path): continue
        with open(anno_path, mode='r') as f:
            anno_lines = f.read()
        anno_lines = anno_lines.split('\n')
        line_meta = {
            'value':
                {
                    'points': [],
                    "closed": True,
                    "polygonlabels":
                        [
                            "front9"
                        ]
                },

            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels"
        }
        for line in anno_lines:
            if line == '': continue
            split_str = line.split(' ')

            num_pts = int((len(split_str) - 1) / 2)
            #line_meta['value']['polygonlabels'][0] = cls_map[int(split_str[0])]
            for num in range(num_pts):
                x = float(split_str[1 + num * 2]) * 100
                y = float(split_str[1 + num * 2 + 1]) * 100
                pts = [x, y]
                line_meta['value']['points'].append(pts)
        task['predictions'][0]['result'].append(line_meta)
        if export_version is not None:
            task['predictions'][0]['model_version'] = export_version
        project.import_tasks(task)
    if export_version is not None:
        project.create_annotations_from_predictions(export_version)


'''
def push_prediction_to_label_studio(proj_id, anno_dir, cls):

    project = ls.get_project(proj_id)
    task_ids = project.get_tasks()

    list_anno = get_list_file_in_folder(anno_dir, ext=['txt'])
    list_anno = [f.replace('.txt','') for f in list_anno]

    for task_id in task_ids:
        task = project.get_task(task_id)
        img_basename = os.path.basename(task['data']['image'])
        img_name = '.'.join(img_basename.split('.')[:-1])
        if img_name in list_anno:
            preds = task['predictions'][0]['result']
            for pred in preds:
            project.create_prediction(
                task_id,
                result=[{
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels",
                    "value": {
                        "closed": True,
                        "points": coord,
                        "polygonlabels": [cls_map[int(label)]]
                    }
                }],

                project.create_prediction(task_ids[0],model_version='1.1')
    project.create_annotations_from_predictions(model_versions='1.1')
    '''
import shutil
from common import rotate_by_exif_metadata, resize_normalize
import numpy as np

def move_imgs_and_anno(img_dir, anno_dir, ok_anno_dir, ok_img_dir):
    list_ok_anno = get_list_file_in_folder(ok_anno_dir, ext=['.txt'])
    for idx, anno in enumerate(list_ok_anno):
        print(idx, anno)
        # os.remove(os.path.join(anno_dir,anno))
        img_name = anno.replace('.txt', '.jpg')
        # shutil.move(os.path.join(img_dir,img_name),os.path.join(ok_img_dir, img_name))
        os.remove(os.path.join(img_dir, img_name))

def filter_data(img_dir, anno_dir):
    list_files = get_list_file_in_folder(img_dir)

    for idx, f in enumerate(list_files):
        print(idx, f)
        img=cv2.imread(os.path.join(img_dir, f))
        anno=cv2.imread(os.path.join(anno_dir, f.replace('.jpg','.png')))
        if img.shape[0]!=anno.shape[0] or img.shape[1]!=anno.shape[1]:
            print('error',100*'-')
            #img_pil = rotate_by_exif_metadata(img_path)
            img = np.array(Image.open(os.path.join(img_dir, f)))
            #cv_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
            img, _= resize_normalize(img, 800)
            #cv_img, _= resize_normalize(cv_img, 800)
            anno, _= resize_normalize(anno, 800)
            cv2.imshow('img', img)
            #cv2.imshow('cv_img', cv_img)
            cv2.imshow('anno', anno)
            cv2.waitKey(0)


if __name__ == '__main__':
    '''
    proj_id = '80'
    img_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/trainval/imgs'
    anno_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/trainval/anno'
    server_img_dir = '/data/local-files/?d=label-studio/data/external/idcard_segment'
    push_yolo_dataset_to_label_studio(proj_id,
                                      img_dir,
                                      anno_dir,
                                      server_img_dir,
                                      cls_file=None,
                                      export_version=None)
    '''

    # anno_dir ='/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/trainval/anno2'
    # push_prediction_to_label_studio(proj_id, anno_dir, cls = ['back9','back12','backchip','front9','front12','frontchip'])




    filter_data('/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg1234/images',
                '/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/eKYC_doc_seg1234/labels')