'''
Replace ancestor patches (non-overlapping) of size 16, 32, 56 or 112 with patches from the adversarial images one at a time.
Check the behaviour of the CNNs when fed with all intermediary images. 
'''

# general
import os
import time
import random
from random import shuffle
import math
import sys
import numpy as np
import copy

# image loading
from PIL import Image
import cv2

# torch
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.rand(1, device="cuda"))
torch.cuda.empty_cache()
from torchvision import transforms
from utils import create_torchmodel, softmax, prediction_preprocess

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

#------------------------------------------------------------------------------------------------------------------------------
def run_network(model, images):
    with torch.no_grad():
        images_copy = images.copy()
        preprocessed_images = torch.from_numpy(images_copy.reshape(1,3,224,224)).type(torch.FloatTensor).to(device)
        preds = model(preprocessed_images).cpu().detach().numpy()
    preds_softmax = softmax(preds) 
    return preds_softmax

def add_patch(network_name,orig,adv,patch_id,comb):
    print(patch_id)
    row = comb[patch_id][0]
    col = comb[patch_id][1]
    patched = copy.deepcopy(orig)
    patched[:,row-half:row+half,col-half:col+half] = adv[:,row-half:row+half,col-half:col+half]
    return patched

# params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
attack_name = 'EA'
data_path = params.data_path
results_path = params.results_path
m=[]
for network in networks:
	m.append(create_torchmodel(network))
	
# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i,model in enumerate(m):
	network = networks[i]

	for patch_size in [16,32,56,112]:
		half = int(patch_size/2)
		comb = np.load(data_path+"/patch_replacement_combs/patch_size"+str(patch_size)+".npy")

		for name in names[3:4]:
			for order in range(1,11):

				print(network,name,patch_size)
				or_class = class_dict[name][0]
				target_class = class_dict[name][1]
				p_orig = []
				p_target = []

				# Get original and adversarial images
				ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
				ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
				ancestor = ancestor.astype(np.uint8)
				ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()
				results_loc = base_path + "results"
				filename = results_loc + "/{}/{}/attack/{}/image{}.npy".format(attack_name,network,name,str(order))
				if os.path.exists(filename):
					adv = np.load(filename)

					for patch_id in range(len(comb)):
						# Add patch
						patched = add_patch(network,ancestor,adv,patch_id,comb)

						# predict using pre-trained network
						pred = run_network(model,patched.copy())

						# save c_a & c_t probs		
						p_orig.append(pred[0,or_class])
						p_target.append(pred[0,target_class])

					# save results
					filename_save_orig = results_loc + "/{}/{}/patch_replacement_nonoverlapping_single/patch_size{}/{}/orig{}.npy".format(attack_name,network,patch_size,name,order)
					filename_save_target = results_loc + "/{}/{}/patch_replacement_nonoverlapping_single/patch_size{}/{}/target{}.npy".format(attack_name,network,patch_size,name,order)
					np.save(filename_save_orig,p_orig)
					np.save(filename_save_target,p_target)
