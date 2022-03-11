'''
Send one CNN's adversarial image to another CNN and check if the other CNN is also fooled
'''
import os
import random
import sys
import numpy as np
import torch
import random
from random import shuffle
import cv2
from PIL import Image

import torch
from torchvision import models

from utils import create_torchmodel, softmax, prediction_preprocess

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

attack_type = sys.argv[1] #'EA' or 'BIM'
shuffled = False
s = 16 #shuffle size
log_file = results_path+'/{}/transferability.log'.format(attack_type)
target_CNN = bagnet17
target_CNN_name = 'BagNet17'
count=0

# Main
#------------------------------------------------------------------------------------------------------------------------------------------
for id_, network in enumerate(list(set(networks)-set([target_CNN_name]))):
	model = m[id_]
	per_network = []
	for name in names:
		for order in range(1,11):
			#print(f"Attack {attack_type} Name {name} Network {network} Order {order}\n")

			or_class = class_dict[name][0]
			target_class = class_dict[name][1]
			pred_class = target_class

			if os.path.exists(results_path + '/{}/{}/attack/{}/image{}.npy'.format(attack_type,network,name,order)):
				if shuffled:
					im = np.load(results_path+"/{}/{}/shuffle_network/{}/{}/images/adv_network{}.npy".format(attack_type,network,s,name,str(order)))
					ancestor = np.load(results_path+"/{}/{}/shuffle_network/{}/{}/images/ancestor{}.npy".format(attack_type,network,s,name,str(order)))
				else:
					im = torch.from_numpy(np.load(results_path+"/{}/{}/attack/{}/image{}.npy".format(attack_type,network,name,order))).float().to('cuda')			
					ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
					ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
					ancestor = ancestor.astype(np.uint8)
					ancestor = prediction_preprocess(Image.fromarray(ancestor.astype(np.uint8))).to('cuda')

				pred = softmax(target_CNN(im.float()).cpu().detach().numpy())[0,pred_class] - softmax(target_CNN(ancestor).cpu().detach().numpy())[0,pred_class]
				per_network.append(pred)

#write to log
with open(log_file,'a') as f:
	f.write('\n'+"{:e}".format(np.array(per_network).mean()))			
