import cv2, os
from PIL import Image, ExifTags

def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    file_names = sorted(file_names)
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

def resize_normalize(img, normalize_width=1000, interpolate = True):
    w = img.shape[1]
    h = img.shape[0]
    interpolate_mode = cv2.INTER_CUBIC if interpolate else cv2.INTER_NEAREST
    if w>normalize_width:
        resize_ratio = normalize_width / w
        normalize_height = round(h * resize_ratio)
        resize_img = cv2.resize(img, (normalize_width, normalize_height), interpolation=interpolate_mode)
        return resize_img, resize_ratio
    else:
        return img, 1.0


def rotate_by_exif_metadata(filepath: str) -> Image:
    """
    xoay lại ảnh theo exif metadata
    nếu load ảnh từ đường dẫn bằng opencv thì thông tin metadata sẽ bị mất
    :param filepath: đường dẫn ảnh
    :return:
    """
    try:
        image = Image.open(filepath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
            elif exif[orientation] == 1:
                print('no rotation!')

        # image.save(filepath)
        # image.close()
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image
