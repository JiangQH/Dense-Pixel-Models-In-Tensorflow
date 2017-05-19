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


incode_csv('/home/qinhong/project/semantic-segmentation-proposals/data/rgb', '/home/qinhong/project/semantic-segmentation-proposals/data/normal_map', 'train.csv')
