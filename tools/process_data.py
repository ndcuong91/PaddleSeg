import os, cv2
from common import get_list_file_in_folder
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

if __name__=='__main__':
    input_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/esign_ekyc_data/testset_1000/warp_imgs/ok'
    output_dir = '/home/misa/PycharmProjects/MISA.eKYC2/data/esign_ekyc_data/testset_1000/warp_imgs/test'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    create_doc_rot_test(input_dir, output_dir)



