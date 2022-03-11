'''
Send both shuffled and unshuffled adversarial images to other CNNs and check if they other CNNs are fooled more with shuffled or unshuffled images
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

from utils import create_torchmodel, softmax, prediction_preprocess

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

networks = params.networks
class_dict = params.class_dict
names = params.names
data_path = params.data_path
results_path = params.results_path
m=[]
for network in networks:
	m.append(create_torchmodel(network))

attack_types = ['EA','BIM']

s1,s2,s3,s4 = 16,32,56,112
res1,res2,res3,res4,res = {},{},{},{},{}

for attack_type in attack_types:
  results_loc = results_path+'/results/{}'.format(attack_type)
  log_file = results_path+'/results/{}/transferability_shuffled.log'.format(attack_type)
	for i, network in enumerate(networks):
		preds1,preds2,preds3,preds4,preds,total_quantity = 0,0,0,0,0,0
		for name in names:
			#comparisons = []	
			print(f"Attack {attack_type} Network {network} Name {name}\n")
			or_class = class_dict[name][0]
			target_class = class_dict[name][1]
			for order in range(1,11):
				if os.path.exists(results_loc + '/{}/attack/{}/image{}.npy'.format(network,name,order)):

					im = torch.from_numpy(np.load(results_loc + '/{}/attack/{}/image{}.npy'.format(network,name,order)).astype('float32')).to('cuda')		
					shuffledim1 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/adv_network{}.npy".format(network,s1,name,order)).astype('float32')).to('cuda')
					shuffledim2 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/adv_network{}.npy".format(network,s2,name,order)).astype('float32')).to('cuda')
					shuffledim3 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/adv_network{}.npy".format(network,s3,name,order)).astype('float32')).to('cuda')
					shuffledim4 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/adv_network{}.npy".format(network,s4,name,order)).astype('float32')).to('cuda')

					ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
					ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
					ancestor = ancestor.astype(np.uint8)
					ancestor = prediction_preprocess(Image.fromarray(ancestor.astype(np.uint8))).to('cuda')
					shuffled_ancestor1 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/ancestor{}.npy".format(network,s1,name,order)).astype('float32')).to('cuda')
					shuffled_ancestor2 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/ancestor{}.npy".format(network,s2,name,order)).astype('float32')).to('cuda')
					shuffled_ancestor3 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/ancestor{}.npy".format(network,s3,name,order)).astype('float32')).to('cuda')
					shuffled_ancestor4 = torch.from_numpy(np.load("/{}/shuffle_network/{}/{}/images/ancestor{}.npy".format(network,s4,name,order)).astype('float32')).to('cuda')

					for j in range(10):
						if j != i:
							model = m[j]
							preds = softmax(model(im).cpu().detach().numpy())[0,target_class] - softmax(model(ancestor).cpu().detach().numpy())[0,target_class]

							preds1 += (softmax(model(shuffledim1).cpu().detach().numpy())[0,target_class] - softmax(model(shuffled_ancestor1).cpu().detach().numpy())[0,target_class]) > preds
							preds2 += (softmax(model(shuffledim2).cpu().detach().numpy())[0,target_class] - softmax(model(shuffled_ancestor2).cpu().detach().numpy())[0,target_class]) > preds
							preds3 += (softmax(model(shuffledim3).cpu().detach().numpy())[0,target_class] - softmax(model(shuffled_ancestor3).cpu().detach().numpy())[0,target_class]) > preds
							preds4 += (softmax(model(shuffledim4).cpu().detach().numpy())[0,target_class] - softmax(model(shuffled_ancestor4).cpu().detach().numpy())[0,target_class]) > preds

							total_quantity += 1
		res1[network] = preds1/total_quantity
		res2[network] = preds2/total_quantity
		res3[network] = preds3/total_quantity
		res4[network] = preds4/total_quantity
		res[network] = preds/total_quantity

# write to log
with open(log_file,'a') as f:
	f.write('\n'+ ' & '.join(str(res1[network]) for network in res1.keys()))
	f.write('\n'+ ' & '.join(str(res2[network]) for network in res2.keys()))
	f.write('\n'+ ' & '.join(str(res3[network]) for network in res3.keys()))
	f.write('\n'+ ' & '.join(str(res4[network]) for network in res4.keys()))
	f.write('\n'+ ' & '.join(str(res[network]) for network in res.keys()))
print(res1,'\n',res2,'\n',res3,'\n',res4)
f.close()
		
