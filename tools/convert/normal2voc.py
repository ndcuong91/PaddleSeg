
import cv2, os
import numpy as np



def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG','jpeg','JPEG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def order_points(pts, vertical=False):
    '''
    Sắp xếp lại các điểm phục vụ cho transformation, trong trường hợp thẻ dọc thì sẽ sắp xếp khác thẻ ngang
    :param pts:
    :param vertical: thẻ dọc
    :return:
    '''
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    if vertical:
        rect[3] = pts[np.argmin(s)]
        rect[1] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(diff)]
        rect[2] = pts[np.argmax(diff)]
    else:
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
    return rect

def find_quadrilateral(img_ori, mask, sw, sh, debug=False, max_object=1):
    '''
    Tìm tứ giác bao quanh mask đã được segment
    :param img_ori: anh goc
    :param mask:
    :param sw: scale width
    :param sh: scale height
    :param debug:
    :param max_object: Số tứ giác tối đa detect được
    :return:
    '''
    list_quads = []
    img = mask
    size_min = min(img.shape[0], img.shape[1])
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, he = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, he = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    list_contours = [f for f in contours]
    list_contours.sort(key=lambda x: x.size, reverse=True)
    list_contours = list_contours[:max_object]

    for cnt in list_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        #print('w', w * sw, ', h', h * sh)
        if (w > size_min / 5 and h > size_min / 5):
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            list_point = approx[:, 0]
            list_point = list_point.astype('float32')
            for p in list_point:
                p[0] *= sw
                p[1] *= sh

            vertical = False
            if w * sw < h * sh: vertical = True
            quad = order_points(list_point, vertical=vertical)  # Sort lại rect theo thứ tự left top, right top...
            quad = quad.astype('int')
            list_quads.append(quad)
            if debug:
                for i in range(0, quad.shape[0] - 1):
                    cv2.line(img_ori, (quad[i][0], quad[i][1]),
                             (quad[i + 1][0], quad[i + 1][1]), (0, 0, 255), 3)
                cv2.line(img_ori, (quad[0][0], quad[0][1]),
                         (quad[quad.shape[0] - 1][0], quad[quad.shape[0] - 1][1]), (0, 0, 255), 3)
                img_show = cv2.resize(img_ori, (800, 800))
                cv2.imshow('res', img_show)
                cv2.imshow('img ori', img_ori)
                cv2.waitKey(0)

            #if len(list_point) != 4:
             #   print('find_quadrilateral. Failed!')
              #  return None

    list_quads.sort(key=lambda x: (x[0][0], x[0][1]))
    return list_quads

def convert_dir(input_dir, output_dir):
    '''

    :param input_dir: contains "imgs" and "anno" directories. in "anno", all annotations should have
    :param output_dir:
    :return:
    '''
    src_img_dir = os.path.join(input_dir, 'imgs')
    src_anno_dir = os.path.join(input_dir, 'anno')
    list_files = get_list_file_in_folder(src_anno_dir)
    for idx, f in enumerate(list_files):
        #if idx <729: continue
        print(idx, f)
        anno_img = cv2.imread(os.path.join(src_anno_dir, f), 0)
        img_path = os.path.join(src_img_dir, f.replace('.png','.jpg'))
        if os.path.exists(img_path):
            img= cv2.imread(img_path)

            list_quads = find_quadrilateral(img, anno_img,1.0,1.0,max_object=5, debug=False)
            voc_line=[]
            w,h = img.shape[1],img.shape[0]
            if list_quads is not None:
                for quad in list_quads:
                    line =[1]
                    for x, y in quad:
                        line.append( round(x/w,2))
                        line.append( round(y/h,2))
                    line = ' '.join([str(f) for f in line])
                    voc_line.append(line)
                voc_line = '\n'.join(voc_line)
                with open(os.path.join(output_dir,f.replace('.png','.txt')), mode = 'w', encoding='utf-8') as fi:
                    fi.write(voc_line)
            else:
                print(idx,f)

if __name__=='__main__':
    input_dir ='/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/trainval'
    output_dir='/home/misa/PycharmProjects/MISA.eKYC2/data/idcard_segment/trainval/anno_yolo'
    convert_dir(input_dir=input_dir,
                output_dir=output_dir)


