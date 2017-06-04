import os
import os.path as osp
import random

def incode_csv(image_dir, label_dir, out_name):
    image_files = os.listdir(image_dir)
    save_names = []
    for img in image_files:
        name, ext = osp.splitext(img)
        label_file = osp.join(label_dir, name+'.png')
        image_file = osp.join(image_dir, img)
        if osp.exists(label_file):
            save_names.append((image_file, label_file))

    random.shuffle(save_names)

    with open(out_name, 'w') as f:
        for (img_file, label_file) in save_names:
            f.write("%s,%s"%(img_file, label_file))
            f.write("\n")
        f.close()
    print 'saving done! as {}'.format(out_name)

def generate_cityscape(image_path, label_path, out_name):
    cities = os.listdir(image_path)
    save_names = []
    for city in cities:
        city_image_dir = osp.join(image_path, city)
        city_label_dir = osp.join(label_path, city)
        images = os.listdir(city_image_dir)
        for image in images:
            name, ext = osp.splitext(image)
            parts = name.split('_')
            parts[-1] = 'gtFine'
            parts.append('labelTrainIds')
            label_base_name = parts[0]
            for i in range(1, len(parts)):
                label_base_name = label_base_name + '_' + parts[i]
            label_base_name += '.png'

            image_file = osp.join(city_image_dir, image)
            label_file = osp.join(city_label_dir, label_base_name)
            if osp.exists(label_file):
                save_names.append((image_file, label_file))

    random.shuffle(save_names)
    with open(out_name, 'w') as f:
        for (im_file, label_file) in save_names:
            f.write("%s,%s"%(im_file, label_file))
            f.write("\n")
        f.close()
    print 'saving done! as {}'.format(out_name)


#incode_csv('/home/jqh/workdir/project/segnet_depth/data/mini/rgb', '/home/jqh/workdir/project/segnet_depth/data/mini/normal_map', 'trial_nyu.csv')