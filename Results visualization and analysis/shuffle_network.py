import numpy as np
import matplotlib.pyplot as plt
from utils import _01
import params

# params
#-------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
results_path = params.results_path

#-------------------------------------------------------------------------------------------------------------------------------------------------
def show_preds(attack_type,patch_size):
    for network in networks:
        for name in names:
            print(f"{network} {name}:\n")
            adv = np.load(results_path+'\\{}\\{}\\shuffle_network\\{}\\{}\\preds\\adv_network.npy'.format(attack_type,network,str(patch_size),name))
            or_, target = class_dict[name][0], class_dict[name][1]
            print(f" Ancestor & target classes, max pred: {adv[0,or_]},{adv[0,target]}, {decode_predictions(adv.reshape(1,1000))[0][0][1]}")

def show_images(patch_size):
    for network in networks:
        print(f"{network}:\n")
        for name in names:
            print(f"{network} {name}:\n")
            adv = np.load(results_path+'\\{}\\{}\\shuffle_network\\{}\\{}\\images\\adv_network.npy'.format(attack_type,network,str(patch_size),name))
            or_, target = class_dict[name][0], class_dict[name][1]
            plt.imshow(_01(adv))
            plt.show()