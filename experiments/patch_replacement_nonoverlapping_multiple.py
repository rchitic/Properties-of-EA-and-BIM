'''
Replace ancestor (non-overlapping) patches of size 8, 16, 32, 56 or 112 with patches from the adversarial images in an adjacent manner.
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

# own
import params
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

def add_patch(network_name,orig,adv,patch_id,comb,patched):
    print(patch_id)
    row = comb[patch_id][0]
    col = comb[patch_id][1]
    patched[:,row-half:row+half+1,col-half:col+half+1] = adv[:,row-half:row+half+1,col-half:col+half+1]
    return patched

# params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
data_path = params.data_path
results_path = params.results_path

SIN = False
if SIN:
	networks = ['ResNet50_SIN']
	resnet50_SIN = create_torchmodel('ResNet50_SIN')
	m = [resnet50_SIN]
else:
	networks = ['VGG16','VGG19','ResNet50','ResNet101','ResNet152','DenseNet121','DenseNet169','DenseNet201','MobileNet','MNASNet']
	vgg16 = create_torchmodel('VGG16')
	vgg19 = create_torchmodel('VGG19')
	resnet50 = create_torchmodel('ResNet50')
	resnet101 = create_torchmodel('ResNet101')
	resnet152 = create_torchmodel('ResNet152')
	densenet121 = create_torchmodel('DenseNet121')
	densenet169 = create_torchmodel('DenseNet169')
	densenet201 = create_torchmodel('DenseNet201')
	mobilenet = create_torchmodel('MobileNet')
	mnasnet = create_torchmodel('MNASNet')
	m = [vgg16,vgg19,resnet50,resnet101,resnet152,densenet121,densenet169,densenet201,mobilenet,mnasnet]

attack_name = 'EA'

# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i,model in enumerate(m):
	network = networks[i]

	for patch_size in [8,16,32,56,112]:
		half = int(patch_size/2)
		comb = np.load(base_path + "code/patch_replacement_combs/patch_size"+str(patch_size)+".npy")

		for name in names:
			or_class = class_dict[name][0]
			target_class = class_dict[name][1]
			p_orig = []
			p_target = []

			# Get ancestor and adversarial images
			ancestor = cv2.imread(data_path+'/imagenet_{}/{}/{}.jpg'.format(name,name,name)) #BGR image
			ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
			ancestor = ancestor.astype(np.uint8)
			ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()
			filename_load = results_path + "/{}/{}/attack/{}/image.npy".format(attack_name,network,name)
			adv = np.load(filename_load)

			# Initialize patched image
			patched = copy.deepcopy(ancestor)

			print(network,name,patch_size)
			for patch_id in range(len(comb)):
				# Add patch
				patched = add_patch(network,ancestor,adv,patch_id,comb,patched)

				# predict using pre-trained network
				pred = run_network(model,patched.copy())

				# save c_a & c_t probs
				p_orig.append(pred[0,or_class])
				p_target.append(pred[0,target_class])

			# save results
			filename_save_orig = results_path + "/{}/{}/patch_replacement_nonoverlapping_full/patch_size{}/{}/orig.npy".format(attack_name,network,patch_size,name)
			filename_save_target = results_path + "/{}/{}/patch_replacement_nonoverlapping_full/patch_size{}/{}/target.npy".format(attack_name,network,patch_size,name)
			np.save(filename_save_orig,p_orig)
			np.save(filename_save_target,p_target)