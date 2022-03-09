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

# own
import params
from utils import create_torchmodel, prediction_preprocess, softmax

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
data_path = params.data_path
results_path = params.results_path

m=[]
for network in networks:
	m.append(create_torchmodel(network))

activations_allImages = {}
image_type = sys.argv[1] #EA or BIM

file_path_ancestor_activations = results_path + '/{}/activations_total_ancestor.pickle'.format(image_type)
quartile = '1'
file_path_adversarial_activations = results_path + '/{}/activations_stats_quartile{}_complete{}.pickle'.format(image_type,quartile,image_type)

if os.path.exists(file_path_adversarial_activations):
	activations_allImages = pickle.load(open(file_path_adversarial_activations,'rb'))
else:
	activations_allImages = {}

if os.path.exists(file_path_ancestor_activations):
	ancestor_total = pickle.load(open(file_path_ancestor_activations,'rb'))
else:
	ancestor_total = get_activation_ancestors()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_activation_ancestor(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().detach().numpy()
    return hook

def get_activation_adversarial(name,object_name,network):
	def hook(model, input, output):
		# get difference in activation between ancestor and adversarial
		activs = output.detach().cpu().detach().numpy()
		diffs = activs - ancestor_total[object_name][network][name]
        
		# get order of magnitude of % change
		om = np.log10(np.abs((activs - ancestor_total[object_name][network][name]) / ancestor_total[object_name][network][name])).astype(int)

		# separate positive / negative ancestor activations into quartiles
		small_vals_neg = (np.abs(ancestor_total[object_name][network][name])>np.quantile(np.abs(ancestor_total[object_name][network][name]),0.5)) & (np.abs(ancestor_total[object_name][network][name])<np.quantile(np.abs(ancestor_total[object_name][network][name]),0.75)) & (ancestor_total[object_name][network][name] < 0)
		small_vals_pos = (np.abs(ancestor_total[object_name][network][name])>np.quantile(np.abs(ancestor_total[object_name][network][name]),0.5)) & (np.abs(ancestor_total[object_name][network][name])<np.quantile(np.abs(ancestor_total[object_name][network][name]),0.75)) & (ancestor_total[object_name][network][name] > 0)

		# given quartile range for ancestor activations, for each oom get:
		# QUARTILE NEG_INITIAL NEG_CHANGE, QUARTILE NEG_INITIAL POS_CHANGE, QUARTILE POS_INITIAL NEG_CHANGE, QUARTILE POS_INITIAL POS_CHANGE 
		stats_list = {}
		for i in range(-1,2):
			loc_om = om == i
			stats_list[i] = ((loc_om * (diffs<0) * small_vals_neg).sum()/small_vals_neg.sum(),(loc_om * (diffs>0) * small_vals_neg).sum()/small_vals_neg.sum(),(loc_om * (diffs<0) * small_vals_pos).sum()/small_vals_pos.sum(),(loc_om * (diffs>0) * small_vals_pos).sum()/small_vals_pos.sum())
		activation[name] = stats_list
	return hook

# ancestor images
def get_activation_ancestors():
	for object_name in names:
		activations_perImage = {}
		ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
		ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
		ancestor = ancestor.astype(np.uint8)
		ancestor = prediction_preprocess(Image.fromarray(ancestor)).to.cuda()

		for i, model in enumerate(m):
			network = networks[i]
			print(object_name,network)
			activation = {}
			hooks = {}
			for module_name, module in model.named_modules():
				if isinstance(module,nn.Conv2d):
					print(module_name)
					hooks[module_name] = module.register_forward_hook(get_activation_ancestor(module_name))
			output = model(ancestor)
			activations_perImage[network] = activation
		activations_allImages[object_name] = activations_perImage

	with open(results_path+"/EA/activations_total_ancestor.pickle", "wb") as dict_file:
		pickle.dump(activations_allImages, dict_file)
	dict_file.close()
	return activations_allImages


# adversarial images
def get_activation_adversarials():
    for object_name in names:
        activations_perImage = {}
        for i, model in enumerate(m):
            print(i)
            activation = {}
            activation_ancestor = {}
            network = networks[i]
            print(object_name,network)
            if shuffle:
                filename_load = results_path + "/{}/{}/shuffle_network/112/{}/images/adv_network.npy".format(image_type,network,object_name)
            else:
                filename_load = results_path + "/{}/{}/attack/{}/image.npy".format(image_type,network,object_name)
            image = torch.from_numpy(np.load(filename_load)).to('cuda').float()
            hooks = {}
            for module_name, module in model.named_modules():
                if isinstance(module,nn.Conv2d):
                    hooks[module_name] = module.register_forward_hook(get_activation_adversarial(module_name,object_name,network))
            output = model(image)
            activations_perImage[network] = list(activation.values())
        activations_allImages[object_name] = activations_perImage

    with open(file_path_adversarial_activations, "wb") as dict_file:
        pickle.dump(activations_allImages, dict_file)
    dict_file.close()



