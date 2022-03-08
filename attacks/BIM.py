'''
Use BIM attack to generate adversarial images for the 10 CNNs.
'''

import numpy as np
import json
import os
import sys
import time

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks

import params
from utils import create_torchmodel, prediction_preprocess

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# Params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
alpha = params.alpha
eps = params.epsilon
N = params.N
data_path = params.data_path
results_path = params.results_path

m = []
for network in networks:
	m.append(create_torchmodel(network))

attack_types = ['BIM']

# Main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for attack_type in attack_types:
	for name in names:
		print(f"Name {name}")
		start = time.time()
		# Get ancestor image

		ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
		ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
		ancestor = ancestor.astype(np.uint8)

		for i,model in enumerate(m):
			network = networks[i]
			print(f"{network}")
			# Set attack parameters
			atk = torchattacks.BIM(model, eps=epsilon, alpha=alpha, steps=N)

			preprocessed_image = prediction_preprocess(Image.fromarray(ancestor))
			target_map_function = lambda preprocessed_image, labels: labels.fill_(class_dict[name][1])
			atk.set_mode_targeted(target_map_function=target_map_function)

			# Create and save adversarial image
			adv_image = atk(preprocessed_image, torch.tensor([4]))
			total_time = time.time()-start
			numpy_adv_image = adv_image.cpu().detach().numpy()
			np.save(results_path + '/{}/{}/attack/{}/image.npy'.format(attack_type,network,name),numpy_adv_image)
			np.save(resuls_path + '/{}/{}/attack/{}/time.npy'.format(attack_type,network,name),total_time)

