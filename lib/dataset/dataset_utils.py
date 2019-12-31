import os


def get_image_list(folder):
    image_list = [os.path.join(folder, x.rsplit('_', 1)[0]) for x in os.listdir(folder) if 'jpg' in x]
    return image_list


if __name__ == '__main__':
    get_image_list('/home/administrator/dataset/helenstar_release/train')