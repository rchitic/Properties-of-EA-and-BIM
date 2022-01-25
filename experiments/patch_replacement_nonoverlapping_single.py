'''
Replace ancestor patches (non-overlapping) of size 8, 16, 32, 56 or 112 with patches from the adversarial images one at a time.
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
from utils import create_torchmodel, create_torchmodel_SIN, softmax, prediction_preprocess

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
class_dict = {'abacus':[398,641],'acorn':[988,947],'baseball':[429,541],'brown_bear':[294,150],'broom':[462,472],'canoe':[472,703],'hippopotamus':[344,368],'llama':[355,340],'maraca':[641,624],'mountain_bike':[671,752]}
names = list(class_dict.keys())

SIN = False
if SIN:
	networks = ['ResNet50_SIN']
	resnet50_SIN = create_torchmodel_SIN('ResNet50_SIN')
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

attack_name = 'EA2'
base_path = "/home/users/rchitic/tvs/"

# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i,model in enumerate(m):
	network = networks[i]

	for patch_size in [8,16,32,56,112]:
		half = int(patch_size/2)
		comb = np.load(base_path+"code/patch_replacement_combs/patch_size"+str(patch_size)+".npy")

		for name in names:
			print(network,name,patch_size)
			or_class = class_dict[name][0]
			target_class = class_dict[name][1]
			p_orig = []
			p_target = []

			# Get original and adversarial images
			ancestor = cv2.imread('/home/users/rchitic/tvs/data/imagenet_{}/{}/{}.jpg'.format(name,name,name)) #BGR image
			ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
			ancestor = ancestor.astype(np.uint8)
			ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()
			results_loc = base_path + "results"
			filename_load = results_loc + "/{}/{}/attack/{}/image.npy".format(attack_name,network,name)
			adv = np.load(filename_load)

			for patch_id in range(len(comb)):
				# Add patch
				patched = add_patch(network,ancestor,adv,patch_id,comb)

				# predict using pre-trained network
				pred = run_network(model,patched.copy())

				# save c_a & c_t probs		
				p_orig.append(pred[0,or_class])
				p_target.append(pred[0,target_class])

			# save results
			filename_save_orig = results_loc + "/{}/{}/patch_replacement_nonoverlapping_single/patch_size{}/{}/orig.npy".format(attack_name,network,patch_size,name)
			filename_save_target = results_loc + "/{}/{}/patch_replacement_nonoverlapping_single/patch_size{}/{}/target.npy".format(attack_name,network,patch_size,name)
			np.save(filename_save_orig,p_orig)
			np.save(filename_save_target,p_target)