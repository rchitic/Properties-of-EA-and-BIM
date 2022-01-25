'''
Use BIM attack to generate adversarial images for the 10 CNNs.
'''

import numpy as np
import time

import cv2
from PIL import Image

import torch
import torchattacks

from utils import create_torchmodel, prediction_preprocess

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# Params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
networks = ['VGG16','VGG19','ResNet50','ResNet101','ResNet152','DenseNet121','DenseNet169','DenseNet201','MobileNet','MNASNet','BagNet9','BagNet17','BagNet33','ResNet50_SIN']
class_dict = {'abacus':[398,641],'acorn':[988,947],'baseball':[429,541],'brown_bear':[294,150],'broom':[462,472],'canoe':[472,703],'hippopotamus':[344,368],'llama':[355,340],'maraca':[641,624],'mountain_bike':[671,752]}
names = list(class_dict.keys())

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
results_loc = '/home/users/rchitic/tvs/results'

# Main
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for attack_type in attack_types:
	for name in names:
		print(f"Name {name}")
		start = time.time()
		# Get ancestor image
		ancestor = cv2.imread('/home/users/rchitic/tvs/data/imagenet_{}/{}/{}.jpg'.format(name,name,name)) #BGR image
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
			np.save(results_loc + '/{}/{}/attack/{}/image.npy'.format(attack_type,network,name),numpy_adv_image)
			np.save(resuls_loc + '/{}/{}/attack/{}/time.npy'.format(attack_type,network,name),total_time)

