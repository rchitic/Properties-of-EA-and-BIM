import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy import signal

from utils import prediction_preprocess, _01
import params

# params
#-------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
rgb = params.rgb
data_path = params.data_path
results_path = params.results_path
analysis_results_path = params.analysis_results_path

# functions
#---------------------------------------------------------------------------------------------------------------------------------------------
# display adversarial images
def show_images(attack_type):
    for network in networks:
        for name in names:
                print(network,name)
                ancestor = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
                ancestor = ancestor.astype(np.uint8)
                ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()

                adv = np.load(results_path+'\\{}\\{}\\attack\\{}\\image.npy'.format(attack_type,network,name))
                if adv.shape == (224,224,3):
                    adv = prediction_preprocess(Image.fromarray(adv.astype(np.uint8))).cpu().detach().numpy()
                
                adv = np.moveaxis(adv,0,-1)
                plt.imsave(analysis_results_path+"\\adversarials\\{}\\{}\\{}.png".format(attack_type,name,network),_01(adv))
                plt.imshow(_01(adv),interpolation='none')
                plt.show()

# display adversarial noise
def show_noise(attack_type): 
    for network in networks:
        for name in names:
            for ch in list(rgb.keys()):
                print(network,name)
                ancestor = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
                ancestor = ancestor.astype(np.uint8)
                ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()

                adv = np.load(results_path+'\\{}\\{}\\attack\\{}\\image.npy'.format(attack_type,network,name))
                if adv.shape == (224,224,3):
                    adv = prediction_preprocess(Image.fromarray(adv.astype(np.uint8))).cpu().detach().numpy()

                plt.imsave(analysis_results_path+"\\noise visualization\\{}\\{}\\{}_{}.png".format(attack_type,network,name,rgb[ch]),_01(adv-ancestor)[ch,:,:])
                plt.imshow(_01(adv-ancestor)[ch,:,:],interpolation='none')
                plt.show()

# display histogram of adversarial noise
def show_hist(attack_type):
    for network in networks:
        for name in names:
            for ch in list(rgb.keys()):
                ancestor = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
                ancestor = ancestor.astype(np.uint8)
                ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()

                adv = np.load(results_path+'\\{}\\{}\\attack\\{}\\image.npy'.format(attack_type,network,name))
                if adv.shape == (224,224,3):
                    adv = prediction_preprocess(Image.fromarray(adv.astype(np.uint8))).cpu().detach().numpy()

                plt.hist((adv-ancestor).flatten(),bins=30)
                plt.title("{} attack, {} {} {}".format(attack_type,network,name,rgb[ch]))
                plt.savefig(analysis_results_path+"\\noise visualization\\{}\\{}\\{}_{}_hist.png".format(attack_type,network,name,rgb[ch]))
                plt.show()

# display autocorrelation of adversarial noise
def show_autocorr(attack_type):
    for network in networks:
        for name in names:
            for ch in list(rgb.keys())[:1]:
                print(network,name)
                ancestor = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
                ancestor = ancestor.astype(np.uint8)
                ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()
                adv = np.load(results_path+'\\{}\\{}\\attack\\{}\\image.npy'.format(attack_type,network,name))
 
                corr = signal.correlate2d((adv-ancestor)[ch,:,:],(adv-ancestor)[ch,:,:], boundary='symm', mode='same')
                plt.imshow(corr,interpolation='none')
                plt.colorbar()
                plt.show()