'''
Replace ancestor (overlapping) patches of size 9, 17 or 33 with patches from the adversarial images continuously.
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

def add_patch(network_name,padded_orig,padded_adv,patch_id):
    print(patch_id)
    row = int(patch_id/224) + half
    col = patch_id%224 + half
    patched = copy.deepcopy(padded_orig)
    patched[:,row-half:row+half+1,col-half:col+half+1] = padded_adv[:,row-half:row+half+1,col-half:col+half+1]
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

attack_name = 'EA'

# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i,model in enumerate(m):
	network = networks[i]
	for patch_size in [17]:
		half = int(patch_size/2)
		for name in names:
			for order in range(1,11):
				print(network,patch_size,name)
				p_orig = []
				p_target = []
				or_class = class_dict[name][0]
				target_class = class_dict[name][1]

				# Get ancestor and adversarial images
				ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
				ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
				ancestor = ancestor.astype(np.uint8)
				ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()

				filename = results_loc + "/{}/{}/attack/{}/image{}.npy".format(attack_name,network,name,order)
				if os.path.exists(filename):			
					adv = np.load(filename)

					# Pad images
					c, x, y = adv.shape
					padded_adv = np.zeros((c,x+patch_size-1,y+patch_size-1))
					padded_adv[:,half:half+x,half:half+y] = adv
					padded_adv = padded_adv.astype(np.float32)

					c, x, y = ancestor.shape
					padded_ancestor = np.zeros((c,x+patch_size-1, y+patch_size-1))
					padded_ancestor[:,half:half+x,half:half+y] = ancestor
					padded_ancestor = padded_ancestor.astype(np.float32)

					for patch_id in range(50176):
						# Add patch
						patched = add_patch(network,padded_ancestor,padded_adv,patch_id)

						# predict using pre-trained network
						pred = run_network(model,patched[:,half:half+224,half:half+224])

						# save c_a & c_t probs
						p_orig.append(pred[0,or_class])
						p_target.append(pred[0,target_class])

					# save results
					filename_save_orig = results_loc + "/{}/{}/patch_replacement_overlapping/patch_size{}/{}/orig{}.npy".format(attack_name,network,patch_size,name,order)
					filename_save_target = results_loc + "/{}/{}/patch_replacement_overlapping/patch_size{}/{}/target{}.npy".format(attack_name,network,patch_size,name,order)
					np.save(filename_save_orig,p_orig)
					np.save(filename_save_target,p_target)
