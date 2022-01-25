'''
Use BagNet-17 to create heatmaps of the c_a & c_t textures 
in the adversarial images. 
'''

# general
import os
import time
import random
from random import shuffle
import math
import sys
import numpy as np

# image loading
from PIL import Image
import cv2

# torch
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#print(torch.rand(1, device="cuda"))
#torch.cuda.empty_cache()
from torchvision import transforms
from utils import create_torchmodel, image_folder_custom_label, Normalize

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

#------------------------------------------------------------------------------------------------------------------------------
# **torch input transformations**
prediction_preprocess = transforms.Compose([
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1], (224,224,3) -> (3,224,224)   
])
norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def run_network(model, images):
    with torch.no_grad():
        preprocessed_images = [prediction_preprocess(Image.fromarray(images[i])) for i in range(len(images))]
        preprocessed_images = torch.stack((preprocessed_images))
        preprocessed_images = preprocessed_images.to(device)
        preds = model(preprocessed_images).cpu().detach().numpy()
    preds_softmax = np.array([softmax(pred) for pred in preds])  
    return preds_softmax

def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()

def generate_heatmap_pytorch(model, image, target_class, or_class, patchsize):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.
    
    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, X, X, 3]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.
    
    """
    import torch
    
    with torch.no_grad():
        # pad with zeros
        _, c, x, y = image.shape
        #preprocessed = prediction_preprocess(Image.fromarray(image[0].astype(np.uint8)))
        preprocessed = torch.from_numpy(image[0]).type(torch.FloatTensor)
        preprocessed = norm_layer(preprocessed)
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize-1)//2:(patchsize-1)//2 + x, (patchsize-1)//2:(patchsize-1)//2 + y] = preprocessed
        image = padded_image[None].astype(np.float32)

        # turn to torch tensor
        input = torch.from_numpy(image).cuda()

        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # compute logits for each patch
        logits_list_target = []
        logits_list_or = []
        it = 0
        print('patches ', patches.shape)
        for patch in patches:
            print(it)
            it = it+1
            with torch.no_grad():
                patch = patch.reshape((1,3,patchsize,patchsize)).cuda()
                logits = model(patch)
                logits_target = logits[:, target_class][:]
                logits_list_target.append(logits_target.data.cpu().numpy().copy())
                logits_or = logits[:, or_class][:]
                logits_list_or.append(logits_or.data.cpu().numpy().copy())
        torch.cuda.empty_cache()

        target = np.hstack(logits_list_target)
        or_ = np.hstack(logits_list_or)
        print('done ', str(patchsize))
        return target, or_

# Get images
SIN = False
if SIN:
	networks = ['ResNet50_SIN']
else:
	networks = ['VGG16','VGG19','ResNet50','ResNet101','ResNet152','DenseNet121','DenseNet169','DenseNet201','MobileNet','MNASNet']
class_dict = {'abacus':[398,641],'acorn':[988,947],'baseball':[429,541],'brown_bear':[294,150],'broom':[462,472],'canoe':[472,703],'hippopotamus':[344,368],'llama':[355,340],'maraca':[641,624],'mountain_bike':[671,752]}
names = list(class_dict.keys())

# Define BagNet models
bagnet9 = create_torchmodel('BagNet9_simple')
bagnet17 = create_torchmodel('BagNet17_simple')
bagnet33 = create_torchmodel('BagNet33_simple')

compute_adv = True
if not compute_adv:
	networks=['']

attack_type = "EA2"

for network in networks:
	for name in names:
		print(network,name)
		or_class = class_dict[name][0]
		target_class = class_dict[name][1]
		
		# Get ancestor and adversarial images
		ancestor = cv2.imread('/home/users/rchitic/tvs/data/imagenet_{}/{}/{}.jpg'.format(name,name,name)) #BGR image
		ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
		ancestor = ancestor.astype(np.uint8)
		results_loc = "/home/users/rchitic/tvs/results/{}".format(attack_type)
				
		if compute_adv: 
			adv = np.load(results_loc + "/{}/attack/{}/image.npy".format(network,name))

			# Get c_a & c_t heatmaps of adversarial images with BagNet9, BagNet17, BagNet33
			t_heatmap_9, o_heatmap_9 = generate_heatmap_pytorch(bagnet9, adv.reshape(1,3,224,224), target_class, or_class, 9)
			np.save(results_loc + "/{}/heatmap/BagNet9/{}/original.npy".format(network,name),o_heatmap_9)
			np.save(results_loc + "/{}/heatmap/BagNet9/{}/target.npy".format(network,name),t_heatmap_9)

			t_heatmap_17, o_heatmap_17 = generate_heatmap_pytorch(bagnet17, adv.reshape(1,3,224,224), target_class, or_class, 17)
			np.save(results_loc + "/{}/heatmap/BagNet17/{}/original.npy".format(network,name),o_heatmap_17)
			np.save(results_loc + "/{}/heatmap/BagNet17/{}/target.npy".format(network,name),t_heatmap_17)
		
			t_heatmap_33, o_heatmap_33 = generate_heatmap_pytorch(bagnet33, adv.reshape(1,3,224,224), target_class, or_class, 33)
			np.save(results_loc + "/{}/heatmap/BagNet33/{}/original.npy".format(network,name),o_heatmap_33)
			np.save(results_loc + "/{}/heatmap/BagNet33/{}/target.npy".format(network,name),t_heatmap_33)

		else:
			# Get c_a & c_t heatmaps of ancestor images with BagNet9, BagNet17, BagNet33
			t_heatmap_9, o_heatmap_9 = generate_heatmap_pytorch(bagnet9, ancestor.reshape(1,224,224,3), target_class, or_class, 9)
			np.save(results_loc + "/heatmap_BagNet9/{}/original.npy".format(name),o_heatmap_9)
			np.save(results_loc + "/heatmap_BagNet9/{}/target.npy".format(name),t_heatmap_9)

			t_heatmap_17, o_heatmap_17 = generate_heatmap_pytorch(bagnet17, ancestor.reshape(1,224,224,3), target_class, or_class, 17)
			np.save(results_loc + "/heatmap_BagNet17/{}/original.npy".format(name),o_heatmap_17)
			np.save(results_loc + "/heatmap_BagNet17/{}/target.npy".format(name),t_heatmap_17)
		
			t_heatmap_33, o_heatmap_33 = generate_heatmap_pytorch(bagnet33, ancestor.reshape(1,224,224,3), target_class, or_class, 33)
			np.save(results_loc + "/heatmap_BagNet33/{}/original.npy".format(name),o_heatmap_33)
			np.save(results_loc + "/heatmap_BagNet33/{}/target.npy".format(name),t_heatmap_33)

