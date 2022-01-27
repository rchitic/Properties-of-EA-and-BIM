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
data_path = params.data_path
results_path = params.results_path

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
bagnet9 = create_torchmodel('BagNet9')
bagnet17 = create_torchmodel('BagNet17')
bagnet33 = create_torchmodel('BagNet33')
resnet50_SIN = create_torchmodel('ResNet50_SIN')

m = [vgg16,vgg19,resnet50,resnet101,resnet152,densenet121,densenet169,densenet201,mobilenet,mnasnet,bagnet9,bagnet17,bagnet33,resnet50_SIN]

attack_types = ['BIM']#'PGD','FGSM','CW'

# Main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for attack_type in attack_types:
	for name in names:
		print(f"Name {name}")
		start = time.time()
		# Get ancestor image
		ancestor = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
		ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
		ancestor = ancestor.astype(np.uint8)
    
		for i,model in enumerate(m):
			network = networks[i]
			print(f"{network}")
			# Set attack parameters
			if attack_type == 'BIM':
				atk = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=7)
			if attack_type == 'CW':
				atk = torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01)
			if attack_type == 'FGSM':
				atk = torchattacks.FGSM(model, eps=8/255)
			if attack_type == 'PGD':
				atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7)

			preprocessed_image = prediction_preprocess(Image.fromarray(ancestor))
			target_map_function = lambda preprocessed_image, labels: labels.fill_(class_dict[name][1])
			atk.set_mode_targeted(target_map_function=target_map_function)

			# Create and save adversarial image
			adv_image = atk(preprocessed_image, torch.tensor([4]))
			total_time = time.time()-start
			numpy_adv_image = adv_image.cpu().detach().numpy()
			np.save(results_path + '/{}/{}/attack/{}/image.npy'.format(attack_type,network,name),numpy_adv_image)
			np.save(resuls_path + '/{}/{}/attack/{}/time.npy'.format(attack_type,network,name),total_time)

