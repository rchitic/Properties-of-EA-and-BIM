'''
Use an EA that, starting from a given ancestor image, generates an adversarial image for a given CNN.
The adversarial image is classified by the CNN as the target class with >= 0.999 probability
The noise is bounded to (-epsilon,epsilon) and only some pixels are mutated, not all
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
from utils import create_torchmodel, image_folder_custom_label, create_torchmodel_SIN

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

#------------------------------------------------------------------------------------------------------------------------------
# **torch input transformations**
prediction_preprocess = transforms.Compose([
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1], (224,224,3) -> (3,224,224)   
])

# **EA functions**
def run_network(model, images):
    with torch.no_grad():
        images_copy = images.copy()
        #preprocessed_images = [prediction_preprocess(Image.fromarray(images_copy[i])) for i in range(len(images_copy))]
        preprocessed_images = [torch.from_numpy(images_copy[i]) for i in range(len(images_copy))]
        preprocessed_images = torch.stack((preprocessed_images)).type(torch.FloatTensor)
        preprocessed_images = preprocessed_images.to(device)

        preds = model(preprocessed_images).cpu().detach().numpy()

    preds_softmax = np.array([softmax(pred) for pred in preds]) 
    return preds_softmax

def get_class_prob(preds, class_no):
    probs = preds[:,class_no]
    return probs

def get_fitness(probs):    
    fitness = probs 
    return fitness

#@nb.njit
def selection(images, fitness):    
    idx_elite = fitness.argsort()[-10:]
    elite_fitness = fitness[idx_elite]
    half_pop_size = images.shape[0]/2
    idx_middle_class = fitness.argsort()[int(half_pop_size):-10]
    elite = images[idx_elite,:][::-1]
    middle_class = images[idx_middle_class,:]

    possible_idx = set(range(0,40)) - set(idx_elite)
    idx_keep = random.sample(possible_idx, 20)
    random_keep = images[idx_keep]

    return elite, middle_class, elite_fitness, idx_elite, random_keep    

# get number of pixels to modify by sampling from a power law distribution 
def get_no_of_pixels(im_size, s, generation):
    if generation == 0:
        np.random.seed(s)
    u_factor = np.random.uniform(0.0,1.0)
    n = 60
    res = (u_factor ** (1.0/(n+1))) * im_size 
    no_of_pixels = im_size - res
    return no_of_pixels

def mutation(no_of_pixels, mutation_group, percentage, s, generation,boundary_min, boundary_max,epsilon,alpha):
    if generation == 0:
        np.random.seed(s)
        random.seed(s)
    mutated_group = mutation_group.copy()
    random.shuffle(mutated_group)
    no_of_individuals = len(mutated_group)

    for individual in range(int(no_of_individuals * percentage)):   
        # select locations to mutate          
        locations_x = np.random.randint(224, size=int(no_of_pixels))
        locations_y = np.random.randint(224, size=int(no_of_pixels))
        locations_z = np.random.randint(3, size=int(no_of_pixels))
        # mutate pixels with +/- alpha
        new_values = random.choices(np.array([-alpha,alpha]),k=int(no_of_pixels))
        mutated_group[individual, locations_z, locations_x, locations_y] = mutated_group[individual, locations_z, locations_x, locations_y] - new_values
        # clip noise noise to +/- epsilon
        noise = mutated_group[individual] - ancestor
        mutated_group[individual] = ancestor + np.clip(noise,-epsilon,epsilon)

    # clip pixel values to [0,1]
    mutated_group = np.clip(mutated_group,boundary_min,boundary_max)    
    return mutated_group      

def get_crossover_parents(crossover_group, s, generation):
    if generation == 0:
        random.seed(s)
    size = crossover_group.shape[0]
    no_of_parents = random.randrange(0, size, 2)
    parents_idx = random.sample(range(0,size), no_of_parents)
    return parents_idx

# select random no of pixels to interchange
def crossover(crossover_group, parents_idx, im_size, s, generation):
    if generation == 0:
        np.random.seed(s)
    crossedover_group = crossover_group.copy()
    no_of_pixels = np.random.randint(im_size)
    for i in range(0, len(parents_idx), 2):
        parent_index_1 = parents_idx[i]
        parent_index_2 = parents_idx[i+1]
        
        size_x = np.random.randint(0, 30)
        start_x = np.random.randint(0, 224- size_x)
        size_y = np.random.randint(0, 30)
        start_y = np.random.randint(0, 224- size_y)
        z = np.random.randint(3)
        
        temp = crossedover_group[parent_index_1, z, start_x : start_x + size_x, start_y : start_y + size_y]
        crossedover_group[parent_index_1, z, start_x : start_x + size_x, start_y : start_y + size_y] = crossedover_group[parent_index_2, z, start_x : start_x + size_x, start_y : start_y + size_y]
        crossedover_group[parent_index_2, z, start_x : start_x + size_x, start_y : start_y + size_y] = temp
        
    return crossedover_group

def softmax(x): 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum()

# **Workflow**
def workflow(or_softmax,target_softmax,iteration,network_name,boundary_min, boundary_max, epsilon,alpha,images, pop_size, im_shape, im_size, class_no, or_class_no, ancestor, ds):

    # create the neural network model
    with torch.no_grad():
        model = create_torchmodel(network_name)

    # parameters 
    percentage_middle_class = 1
    percentage_keep = 1
    probs_softmax = [0] * pop_size
    original_class_prob_plot = []
    fitness = np.array([0]*pop_size)
    s = 30
    start = time.time()

    # repeat until image is confidently adversarial
    while(np.max(probs_softmax) < 0.999):
                        
        # select population classes based on fitness
        elite, middle_class, elite_fitness, idx_elite, random_keep = selection(images, fitness)
        elite2 = elite.copy()
        keep = np.concatenate((elite2,random_keep))
        
        # mutate and crossover individuals
        no_of_pixels = get_no_of_pixels(im_size, s, iteration)
        
        mutated_middle_class = mutation(no_of_pixels, middle_class, percentage_middle_class, s, iteration,boundary_min,boundary_max,epsilon,alpha)
        mutated_keep_group1 = mutation(no_of_pixels, keep, percentage_keep, s, iteration,boundary_min,boundary_max,epsilon,alpha)
        mutated_keep_group2 = mutation(no_of_pixels, mutated_keep_group1, percentage_keep, s, iteration,boundary_min,boundary_max,epsilon,alpha)
        
        all_ = np.concatenate((mutated_middle_class, mutated_keep_group2))
        parents_idx = get_crossover_parents(all_, s, iteration)
        crossover_group = crossover(all_, parents_idx, im_size, s, iteration)

        # create new population 
        images = np.concatenate((elite, crossover_group))   
        iteration = iteration + 1

        # run network with new population 
        preds_softmax = run_network(model, images)

        # get c_a & c_t probs
        probs_softmax = get_class_prob(preds_softmax, class_no)
        target_softmax.append(probs_softmax)
        probs_or_softmax = get_class_prob(preds_softmax, or_class_no)
        or_softmax.append(probs_or_softmax)
        print(f"Target prob softmax {np.max(probs_softmax)}, ancestor prob softmax {probs_or_softmax[0]}")

        # get fitness of individuals
        fitness = get_fitness(probs_softmax)
    
    duration = time.time() - start
    print(f"Total time: {duration/60} min")        
    return images, np.array(or_softmax), np.array(target_softmax), iteration, duration

#----------------------------------------------------------------------------------------------------------------------
# Main 
networks = {1:'VGG16',2:'VGG19',3:'ResNet50',4:'ResNet101',5:'ResNet152',6:'DenseNet121',7:'DenseNet169',8:'DenseNet201',9:'MobileNet',10:'MNASNet',11:'BagNet9',12:'BagNet17',13:'BagNet33',14:'ResNet50_SIN'}
network_ID = int(sys.argv[1])
network_name = networks[network_ID]

class_dict = {'abacus':[398,641],'acorn':[988,947],'baseball':[429,541],'brown_bear':[294,150],'broom':[462,472],'canoe':[472,703],'hippopotamus':[344,368],'llama':[355,340],'maraca':[641,624],'mountain_bike':[671,752]}
ID = int(sys.argv[2])
name = list(class_dict.keys())[ID]
or_class = class_dict[name][0]
target_class = class_dict[name][1]
print(f"Network {network_name} image {name}")

ancestor = cv2.imread('/home/users/rchitic/tvs/data/imagenet_{}/{}/{}.jpg'.format(name,name,name)) #BGR image
ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
ancestor = ancestor.astype(np.uint8)
ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()

pop_size = 40
images = np.array([ancestor]*pop_size).astype(float)
or_softmax = []
target_softmax = []
iteration = 0
im_shape = (224,224,3)
im_size = 224*224*3
ds = 0 # 0 for l2_norm, 1 for structural similarity
boundary_min = 0
boundary_max = 1
epsilon = 8/255
alpha = 2/255

# run the EA
res, or_softmax, target_softmax, iteration, duration = workflow(or_softmax,target_softmax,iteration,network_name, boundary_min, boundary_max, epsilon,alpha,images, pop_size, im_shape, im_size, target_class, or_class, ancestor, ds)

# save results	
file_save = "/home/users/rchitic/tvs/results/EA2/{}/attack/{}/".format(network_name,name)
np.save(file_save+"image.npy", res[0])
np.save(file_save+"or_softmax.npy", or_softmax)
np.save(file_save+"target_softmax.npy", target_softmax)
np.save(file_save+"generations.npy", iteration)
np.save(file_save+"time.npy", duration)
