'''
Shuffle adversarial images with shuffle sizes: 16,32,56,112.
Check how the CNNs classify the shuffled adverarial images.
'''

# general
import os
import time
import random
from random import shuffle
import math
import sys
import numpy as np

# image loading
from PIL import Image
import cv2

# torch
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#print(torch.rand(1, device="cuda"))
#torch.cuda.empty_cache()
from torchvision import transforms
from utils import create_torchmodel, prediction_preprocess, softmax

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_network(model, images):
    with torch.no_grad():
        images_copy = images.copy()
        preprocessed_images = torch.from_numpy(images_copy.reshape(1,3,224,224)).type(torch.FloatTensor).to(device)
        preds = model(preprocessed_images).cpu().detach().numpy()
    preds_softmax = softmax(preds) 
    return preds_softmax

def shuffle(im, comb1, comb2, half):
    locs = []
    shuffled_im = np.zeros((3,224,224))
    for x in range(len(comb1)): 
        bbox_centre1_ij = comb1[x]
        bbox_centre2_ij = comb2[x]
        loc_i = bbox_centre1_ij[0]
        loc_j = bbox_centre1_ij[1]
        loc_i2 = bbox_centre2_ij[0]
        loc_j2 = bbox_centre2_ij[1]
        shuffled_im[:,loc_i2-half:loc_i2+half,loc_j2-half:loc_j2+half] = im[:,loc_i-half:loc_i+half,loc_j-half:loc_j+half]
    return shuffled_im

# params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
data_path = params.data_path
results_path = params.results_path
attack_name = 'EA'
m=[]
for network in networks:
	m.append(create_torchmodel(network))

# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i,model in enumerate(m):
	network = networks[i]
	for shuffle_size in [8,16,32,56,112]:
		half = int(shuffle_size/2)
		comb1 = np.load(data_path+'/shuffle_combs/'+str(shuffle_size)+'/comb1.npy')
		comb2 = np.load(data_path+'/shuffle_combs/'+str(shuffle_size)+'/comb2.npy')

		for name in names:
			for order in range(1,11):
				results_loc = results_path + "/{}".format(attack_name)
				path = results_loc + "/{}/attack/{}/image{}.npy".format(network,name,str(order))
				if os.path.exists(path):
					print(network,shuffle_size,name)
					# Get ancestor and adversarial images
					ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
					ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
					ancestor = ancestor.astype(np.uint8)
					ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()
					adv_network = np.load(path)
			
					# Shuffle images
					ancestor = shuffle(ancestor,comb1,comb2,half)
					adv_network = shuffle(adv_network,comb1,comb2,half)

					image_save_loc = results_loc +"/{}/shuffle_network/{}/{}/images".format(network,shuffle_size,name)
					np.save(image_save_loc + "/ancestor{}.npy".format(str(order)),ancestor)
					np.save(image_save_loc + "/adv_network{}.npy".format(str(order)),adv_network)

					# predict using pre-trained network
					p_ancestor = run_network(model,ancestor)
					p_adv_network = run_network(model,adv_network)
					p_adv_bagnet9 = run_network(bagnet9,adv_network)
					p_adv_bagnet17 = run_network(bagnet17,adv_network)
					p_adv_bagnet33 = run_network(bagnet33,adv_network)

					pred_save_loc = results_loc +"/{}/shuffle_network/{}/{}/preds".format(network,shuffle_size,name)
					np.save(pred_save_loc + "/ancestor{}.npy".format(str(order)),p_ancestor)
					np.save(pred_save_loc + "/adv_network{}.npy".format(str(order)),p_adv_network)
					np.save(pred_save_loc + "/adv_bagnet9{}.npy".format(str(order)),p_adv_bagnet9)
					np.save(pred_save_loc + "/adv_bagnet17{}.npy".format(str(order)),p_adv_bagnet17)
					np.save(pred_save_loc + "/adv_bagnet33{}.npy".format(str(order)),p_adv_bagnet33)
