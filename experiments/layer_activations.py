'''
Get activation of the CNNs' convolutional layers when fed with ancestors & adversarial images.
Divide ancestor activations into quartiles and measure the positive & negative changes undergone by each.
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
import pickle
from scipy.stats import skew
import sys
import os.path

# image loading
from PIL import Image
import cv2

# torch
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#print(torch.rand(1, device="cuda"))
#torch.cuda.empty_cache()
from torchvision import transforms
import torch.nn as nn

from utils import create_torchmodel, prediction_preprocess, softmax

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
class_dict = params.class_dict
names = params.names
networks = params.networks
results_loc = params.results_loc
data_path = params.data_path
shuffle=False
image_type = sys.argv[1] #EA or BIM

m=[]
for network in networks:
	m.append(create_torchmodel(network))

activations_allImages = {}

file_path_ancestor_activations = results_loc + '/{}/activations_total_ancestor.pickle'.format(image_type)
file_path_adversarial_activations = results_loc + '/{}/activations_stats_quartile3_random_complete{}.pickle'.format(image_type,image_type)

if os.path.exists(file_path_adversarial_activations):
	activations_allImages = pickle.load(open(file_path_adversarial_activations,'rb'))
else:
	activations_allImages = {}

def get_activation_ancestor(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().detach().numpy()
    return hook

# ancestor images
def get_activation_ancestors():
	for object_name in names:
		activations_perImage = {}
		for order in range(1,11):
			activations_perOrder = {}
			ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
			ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
			ancestor = ancestor.astype(int)
			ancestor = prediction_preprocess(Image.fromarray(ancestor.astype(np.uint8))).to('cuda')

			for i, model in enumerate(m):
				network = networks[i]
				print(object_name,network)
				activation = {}
				hooks = {}
				for module_name, module in model.named_modules():
					if isinstance(module,nn.Conv2d):
						print(module_name)
						hooks[module_name] = module.register_forward_hook(get_activation_ancestor(module_name,activation))
				output = model(ancestor)
				activations_perOrder[network] = activation
			activations_perImage[order] = activations_perOrder
		activations_allImages[object_name] = activations_perImage

	with open("/home/users/rchitic/tvs/results/EA2/activations_total_ancestor.pickle", "wb") as dict_file:
		pickle.dump(activations_allImages, dict_file)
	dict_file.close()
	return activations_allImages

if os.path.exists(file_path_ancestor_activations):
	ancestor_total = pickle.load(open(file_path_ancestor_activations,'rb'))
else:
	ancestor_total = get_activation_ancestors()

def get_activation_adversarial(name,object_name,order,network):
	def hook(model, input, output):
		# get difference in activation between ancestor and adversarial
		activs = output.detach().cpu().detach().numpy()
		diffs = activs - ancestor_total[object_name][order][network][name]
        
		# get order of magnitude of % change
		om = np.log10(np.abs((activs - ancestor_total[object_name][order][network][name]) / ancestor_total[object_name][order][network][name])).astype(int)

		# separate positive / negative ancestor activations into quartiles
		small_vals_neg = (np.abs(ancestor_total[object_name][order][network][name])>np.quantile(np.abs(ancestor_total[object_name][order][network][name]),0.5)) & (np.abs(ancestor_total[object_name][order][network][name])<np.quantile(np.abs(ancestor_total[object_name][order][network][name]),0.75)) & (ancestor_total[object_name][order][network][name] < 0)
		small_vals_pos = (np.abs(ancestor_total[object_name][order][network][name])>np.quantile(np.abs(ancestor_total[object_name][order][network][name]),0.5)) & (np.abs(ancestor_total[object_name][order][network][name])<np.quantile(np.abs(ancestor_total[object_name][order][network][name]),0.75)) & (ancestor_total[object_name][order][network][name] > 0)

		# given quartile range for ancestor activations, for each oom get:
		# QUARTILE NEG_INITIAL NEG_CHANGE, QUARTILE NEG_INITIAL POS_CHANGE, QUARTILE POS_INITIAL NEG_CHANGE, QUARTILE POS_INITIAL POS_CHANGE 
		stats_list = {}
		for i in range(-1,2):
			loc_om = om == i
			stats_list[i] = ((loc_om * (diffs<0) * small_vals_neg).sum()/small_vals_neg.sum(),(loc_om * (diffs>0) * small_vals_neg).sum()/small_vals_neg.sum(),(loc_om * (diffs<0) * small_vals_pos).sum()/small_vals_pos.sum(),(loc_om * (diffs>0) * small_vals_pos).sum()/small_vals_pos.sum())
		activation[name] = stats_list
	return hook

# adversarial images
for object_name in names:
	activations_perImage = {}
	for order in [1,4]:
		activations_perOrder = {}
		for i, model in enumerate(m):
			print(i)
			activation = {}
			activation_ancestor = {}
			network = networks[i]
			print(object_name,network)
			if shuffle:
				filename_load = results_loc + "/{}/{}/shuffle_network/112/{}/images/adv_network{}.npy".format(image_type,network,object_name,order)
			else:
				filename_load = results_loc + "/{}/{}/attack/{}/image{}.npy".format(image_type,'DenseNet121',object_name,order)
			
			if os.path.exists(results_loc + "/{}/{}/attack/{}/image{}.npy".format('EA2',network,object_name,order)):
				image = torch.from_numpy(np.load(filename_load)).to('cuda').float()
				hooks = {}
				for module_name, module in model.named_modules():
					if isinstance(module,nn.Conv2d):
						hooks[module_name] = module.register_forward_hook(get_activation_adversarial(module_name,object_name,order,network))
				output = model(image)
				activations_perOrder[network] = list(activation.values())
		activations_perImage[order] = activations_perOrder
	activations_allImages[object_name] = activations_perImage

with open(file_path_adversarial_activations, "wb") as dict_file:
	pickle.dump(activations_allImages, dict_file)
dict_file.close()
