# general
import os
import math
import sys
import numpy as np
import cv2
from PIL import Image

import torch

# own
from utils import create_torchmodel, softmax, prediction_preprocess
import params

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# params
#--------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
data_path = params.data_path
results_path = params.results_path

bagnet9 = create_torchmodel('BagNet9')
bagnet17 = create_torchmodel('BagNet17')
bagnet33 = create_torchmodel('BagNet33')
resnet50_SIN = create_torchmodel_SIN('ResNet50_SIN')
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

shuffle = False
s = 16 #shuffle size
attack_type = 'EA'
log_file = results_path+'/{}/transferability.log'.format(attack_type)
target_CNN = bagnet17
target_CNN_name = 'BagNet17'
per_network = []

# Main
#------------------------------------------------------------------------------------------------------------------------------------------
for network in list(set(networks)-set([target_CNN_name])):
	for name in names:
		print(f"Attack {attack_type} Network {network} Name {name}\n")

		print(network,'\n')	
		or_class = class_dict[name][0]
		target_class = class_dict[name][1]
		#target_class = or_class
		results_loc = '/home/users/rchitic/tvs/results/{}'.format(attack_type)
			
		if shuffled:
			im = np.load(results_path+"/{}/{}/shuffle_network/{}/{}/images/adv_network.npy".format(attack_type,network,s,name))
			ancestor = np.load(results_path+"/{}/{}/shuffle_network/{}/{}/images/ancestor.npy".format(attack_type,network,s,name))
		else:
			im = torch.from_numpy(np.load(results_path + '/{}/attack/{}/image.npy'.format(network,name))).float().to('cuda')			
			ancestor = cv2.imread(data_path+'/imagenet_{}/{}/{}.jpg'.format(name,name,name)) #BGR image
			ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
			ancestor = ancestor.astype(np.uint8)
			ancestor = prediction_preprocess(Image.fromarray(ancestor.astype(np.uint8))).to('cuda')

		pred = softmax(target_CNN(im.float()).cpu().detach().numpy())[0,target_class] - softmax(target_CNN(ancestor).cpu().detach().numpy())[0,target_class]
		print(pred)		
		per_network.append(pred)

# write to log
with open(log_file,'a') as f:
	f.write('\n'+str(np.array(per_network).mean()))			