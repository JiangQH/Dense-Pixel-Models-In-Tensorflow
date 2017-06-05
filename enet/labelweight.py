# compute class weights for all the class labels
import os.path as osp
import os
import numpy as np
import scipy.misc
import json

Data_dir = '/home/qinhong/project/dataset/cityscape/label/train/'
def compute_lable_weight():
    cities = os.listdir(Data_dir)
    fres = {}
    total_count = 0
    for city in cities:
        city_dir = osp.join(Data_dir, city)
        print 'handling dir {}'.format(city)
        scenes = os.listdir(city_dir)
        for scene in scenes:
            name, ext = osp.splitext(scene)
            ends = name.split('_')[-1]
            if ends == 'labelTrainIds':
                data = scipy.misc.imread(osp.join(city_dir, scene))
                #plt.imshow(data)
                uniques, counts = np.unique(data, return_counts=True)
                for i in range(len(uniques)):
                    if uniques[i] == 19:
                        continue
                    if uniques[i] not in fres:
                        fres[uniques[i]] = 0
                    fres[uniques[i]] += counts[i]
                    total_count += counts[i]
    # save to disk
    probs = []
    for label in fres:
        prob = fres[label] / (total_count * 1.0)
        print '{} : {}'.format(label, prob)
        probs.append(prob)
    with open('cityscape.txt', 'w') as fp:
        json.dump(probs, fp)

if __name__ == '__main__':
    compute_lable_weight()