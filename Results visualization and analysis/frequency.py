import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
from PIL import Image
import copy

import torch
import params
from utils import prediction_preprocess, softmax, create_torchmodel 

# params
#-------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
rgb = params.rgb
data_path = params.data_path
results_path = params.results_path
analysis_results_path = params.analysis_results_path

# Magnitude spectra of ancestors & adversarials
#--------------------------------------------------------------------------------------------------------------------------------------------------
def magnitude_spectra(attack_type,networks,diff):
    for network in networks:
        for name in names:
            for channel in [0,1,2]:   
                rgb_letter = rgb[channel]
                print(network,name,rgb_letter)
                path = results_path+'\\{}\\{}\\attack\\{}\\image.npy'
                adv = np.load(path.format(attack_type,network,name))
                ancestor = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
                ancestor = ancestor.astype(np.uint8)

                print(adv.shape)
                ancestor = prediction_preprocess(Image.fromarray(ancestor))
                ancestor = ancestor.cpu().detach().numpy()
                if diff:
                    calc = 'diff(magn)'
                        
                    # |adv| - |ancestor|
                    f_or = np.fft.fft2(ancestor[channel,:,:])
                    fshiftor = np.fft.fftshift(f_or)
                    magnitude_spectrumor = 20*np.log(np.abs(fshiftor))
                    plt.imshow(magnitude_spectrumor, cmap = 'gray')
                    #plt.show()

                    fadv = np.fft.fft2(adv[channel,:,:])
                    fshiftadv = np.fft.fftshift(fadv)
                    magnitude_spectrumadv = 20*np.log(np.abs(fshiftadv))
                    plt.imshow(magnitude_spectrumadv, cmap = 'gray')
                    #plt.show()
                    
                    plt.imshow(magnitude_spectrumadv-magnitude_spectrumor,cmap='bwr',interpolation='none')
                    plt.title("{} attack, {} {} {} {}".format(attack_type,network,name,rgb_letter,calc))
                    plt.colorbar()
                    plt.savefig(analysis_results_path+"\\frequency\\{}\\{}\\{}_{}_diff(magn).png".format(attack_type,network,name,rgb_letter))
                    plt.show()

                else:
                    calc = 'magn(diff)'                        
                    
                    # |adv-ancestor|
                    f = np.fft.fft2(adv[channel,:,:]-ancestor[channel,:,:])
                    fshift = np.fft.fftshift(f)
                    magnitude_spectrum = 20*np.log(np.abs(fshift))

                    plt.imshow(magnitude_spectrum,cmap='bwr',interpolation='none')
                    plt.title("{} attack, {} {} {} {}".format(attack_type,network,name,rgb_letter,calc))
                    plt.colorbar()
                    plt.savefig(analysis_results_path+"\\frequency\\{}\\{}\\{}_{}_magn(diff).png".format(attack_type,network,name,rgb_letter))
                    plt.show()

# Filtering functions
#--------------------------------------------------------------------------------------------------------------------------------------------------
# Low-pass
import math
def low_pass_filtering(image, radius):
    """
         Low pass filter function
         :param image: input image
         :param radius: radius
         :return: filtering result
    """
         # Fourier transform the image, fft is a three-dimensional array, fft[:, :, 0] is the real part, fft[:, :, 1] is the imaginary part
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
         # Centralize fft, the generated dshift is still a three-dimensional array
    dshift = np.fft.fftshift(fft)
 
         # Get the center pixel
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

        # Build mask, 256 bits, two channels
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[mid_row - radius:mid_row + radius, mid_col - radius:mid_col + radius] = 1
    plt.imshow(mask[:,:,0],interpolation='none')

    # Multiply the Fourier transform result by a mask
    fft_filtering = dshift * mask
         # Inverse Fourier transform
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
         # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
    cv2.normalize(image_filtering, image_filtering, 0, 255, cv2.NORM_MINMAX)
    return image_filtering

def high_pass_filtering(image, radius, n):
    """
         High pass filter function
         :param image: input image
         :param radius: radius
         :param n: ButterWorth filter order
         :return: filtering result
    """
         # Fourier transform the image, fft is a three-dimensional array, fft[:, :, 0] is the real part, fft[:, :, 1] is the imaginary part
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
         # Centralize fft, the generated dshift is still a three-dimensional array
    dshift = np.fft.fftshift(fft)
 
         # Get the center pixel
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
         # Build ButterWorth high-pass filter mask
 
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
                         # Calculate the distance from (i, j) to the center
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            try:
                mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + pow(radius / d, 2*n))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
         # Multiply the Fourier transform result by a mask
    fft_filtering = dshift * mask
         # Inverse Fourier transform
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
         # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
    cv2.normalize(image_filtering, image_filtering, 0, 255, cv2.NORM_MINMAX)
    return image_filtering
 
 
def bandpass_filter(image, radius, w, n=1):
    """
         Bandpass filter function
         :param image: input image
         :param radius: distance from the center of the band to the origin of the frequency plane
         :param w: bandwidth
         :param n: order
         :return: filtering result
    """
         # Fourier transform the image, fft is a three-dimensional array, fft[:, :, 0] is the real part, fft[:, :, 1] is the imaginary part
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
         # Centralize fft, the generated dshift is still a three-dimensional array
    dshift = np.fft.fftshift(fft)
 
         # Get the center pixel
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
         # Build mask, 256 bits, two channels
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
                         # Calculate the distance from (i, j) to the center
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 1
            else:
                mask[i, j, 0] = mask[i, j, 1] = 0
 
         # Multiply the Fourier transform result by a mask
    fft_filtering = dshift * np.float32(mask)
         # Inverse Fourier transform
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
         # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
    cv2.normalize(image_filtering, image_filtering, 0, 255, cv2.NORM_MINMAX)
    return image_filtering

def bandstop_filter(image, radius, w, n=1):
    """
         Bandpass filter function
         :param image: input image
         :param radius: distance from the center of the band to the origin of the frequency plane
         :param w: bandwidth
         :param n: order
         :return: filtering result
    """
         # Fourier transform the image, fft is a three-dimensional array, fft[:, :, 0] is the real part, fft[:, :, 1] is the imaginary part
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
         # Centralize fft, the generated dshift is still a three-dimensional array
    dshift = np.fft.fftshift(fft)
 
         # Get the center pixel
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
         # Build mask, 256 bits, two channels
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
                         # Calculate the distance from (i, j) to the center
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 0
            else:
                mask[i, j, 0] = mask[i, j, 1] = 1
 
         # Multiply the Fourier transform result by a mask
    fft_filtering = dshift * np.float32(mask)
         # Inverse Fourier transform
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
         # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
    cv2.normalize(image_filtering, image_filtering, 0, 255, cv2.NORM_MINMAX)
    return image_filtering

# Graphs
#--------------------------------------------------------------------------------------------------------------------------------------------------
# Prob vs. stopped frequency interval
def bandstop_graph(attack_type):
    for network in networks:
        model = create_torchmodel(network)
        for name in names:      
                print(network,name)
                preds_ancestor_orclass,preds_ancestor_targetclass,preds_adv_orclass,preds_adv_targetclass = [],[],[],[]
                path = results_path+'\\results\\{}\\{}\\attack\\{}\\image.npy'

                adv = np.load(path.format(attack_type,network,name))
                advnochange = copy.deepcopy(adv).astype('float32')
                
                or_ = cv2.imread('C:\\Users\\raluca.chitic\\Desktop\\PhD\\tvs\\data\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                or_ = cv2.resize(or_,(224,224))[:,:,::-1] #RGB image
                or_ = prediction_preprocess(Image.fromarray(or_.astype(np.uint8))).cpu().detach().numpy()
                
                for radius in range(15,115,10):
                    
                    # Ancestor
                    # Low-pass filter with the current radius
                    filtered_r = bandstop_filter(or_[0,:,:],radius,30,1)
                    filtered_g = bandstop_filter(or_[1,:,:],radius,30,1)
                    filtered_b = bandstop_filter(or_[2,:,:],radius,30,1)
                    filtered = np.dstack((filtered_r,filtered_g,filtered_b))
                    #np.save(analysis_results_path+"\\ancestors_frequency_bandstop\\{}\\{}\\{}\\{}.npy".format(attack_type,network,name,radius),filtered)
                    
                    # Get prediction of filtered image
                    with torch.no_grad():
                        model = model.eval()
                        pred_filtered = model(prediction_preprocess(Image.fromarray(filtered.astype(np.uint8))).reshape(1,3,224,224))
                        pred_filtered = softmax(pred_filtered[0].cpu().detach().numpy())
                    preds_ancestor_orclass.append(pred_filtered[class_dict[name][0]])
                    preds_ancestor_targetclass.append(pred_filtered[class_dict[name][1]])
                    
                    # Adversarial
                    # Low-pass filter with the current radius
                    filtered_r = bandstop_filter(adv[0,:,:],radius,30,1)
                    filtered_g = bandstop_filter(adv[1,:,:],radius,30,1)
                    filtered_b = bandstop_filter(adv[2,:,:],radius,30,1)
                    filtered = np.dstack((filtered_r,filtered_g,filtered_b))
                    #np.save("C:\\Users\\raluca.chitic\\Desktop\\PhD\\tvs\\analysis\\results\\adversarials_frequency_bandstop\\{}\\{}\\{}\\{}.npy".format(attack_type,network,name,radius),filtered)

                    # Get prediction of filtered image
                    with torch.no_grad():
                        model = model.eval()
                        pred_filtered = model(prediction_preprocess(Image.fromarray(filtered.astype(np.uint8))).reshape(1,3,224,224))
                        pred_filtered = softmax(pred_filtered[0].cpu().detach().numpy())
                    preds_adv_orclass.append(pred_filtered[class_dict[name][0]])
                    preds_adv_targetclass.append(pred_filtered[class_dict[name][1]])

                    
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))

                ax1.plot(np.log10(preds_ancestor_orclass),label='ancestor')
                ax1.plot(np.log10(preds_adv_orclass),label='adversarial',linestyle='dashed')

                ax2.plot(np.log10(preds_ancestor_targetclass),label='ancestor')
                ax2.plot(np.log10(preds_adv_targetclass),label='adversarial',linestyle='dashed')

                f.suptitle('{} band-stop no shuffle'.format(attack_type),size=20)
                ax1.set_title(r'$c_a$',size=20)
                ax2.set_title(r'$c_t$',size=20)
                ax1.set_xlabel('Radius',size=20)
                ax2.set_xlabel('Radius',size=20)
                ax1.set_ylabel('Log probability',size=20)
                ax2.set_ylabel('Log probability',size=20)

                ax1.set_xticks(range(10))
                ax2.set_xticks(range(10))
                ax1.set_xticklabels(list(range(15,115,10)),fontsize=15)
                ax2.set_xticklabels(list(range(15,115,10)),fontsize=15)
                ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax1.tick_params(axis='y', which='major', labelsize=15)

                ax1.legend(prop={'size': 15})
                ax2.legend(prop={'size': 15})

                file = analysis_results_path+"\\bandstop_graphs\\{}\\{}\\{}.png".format(attack_type,network,name)
                plt.savefig(file)
                plt.show()

# Prob vs. magnitude of noise
def partial_noise():
    for network in networks:
        model = create_torchmodel(network)
        for name in names:      
                print(network,name)
                attack_type = 'BIM'
                maxBIM2, maxEA2, maxBIM3, maxEA3 = [],[],[],[]
                maxBIM, maxEA, preds_ancestor_or_EA,preds_ancestor_target_EA,preds_ancestor_or_BIM,preds_ancestor_target_BIM = [],[],[],[],[],[]
                target_EA_2,target_EA_3,or_BIM_2,or_BIM_3,target_BIM_2,target_BIM_3,or_EA_2,or_EA_3=[],[],[],[],[],[],[],[]
                preds_ancestor_orclass,preds_ancestor_targetclass,preds_adv_orclass,preds_adv_targetclass = [],[],[],[]
                
                path = results_path+'\\{}\\{}\\attack\\{}\\image.npy'
                advBIM = np.load(path.format('BIM',network,name))      
                path = results_path+'\\{}\\{}\\attack\\{}\\image.npy'
                advEA = np.load(path.format('EA',network,name))                
                or_ = cv2.imread(data_path+'\\imagenet_{}\\{}\\{}.jpg'.format(name,name,name)) #BGR image
                or_ = cv2.resize(or_,(224,224))[:,:,::-1] #RGB image
                ancestor = prediction_preprocess(Image.fromarray(or_.astype(np.uint8))).cpu().detach().numpy()
                
                model2 = create_torchmodel('ResNet152').eval()
                model3 = create_torchmodel('DenseNet201').eval()

                for radius in range(5,305,5):
                    n = advEA-ancestor
                    n = n*radius/100
                    newEA = ancestor + n
                    newEA = newEA.astype('float32')
                    
                    n = advBIM-ancestor
                    n = n*radius/100
                    newBIM = ancestor + n
                    newBIM = newBIM.astype('float32')

                    # BIM: Get prediction of filtered image
                    with torch.no_grad():
                        model = model.eval()
                        pred_filtered = model(torch.from_numpy(newBIM).reshape(1,3,224,224))
                        pred_filtered_BIM = softmax(pred_filtered[0].cpu().detach().numpy())
                    with torch.no_grad():
                        model2 = model2.eval()
                        pred_filtered = model2(torch.from_numpy(newBIM).reshape(1,3,224,224))
                        pred_filtered_BIM2 = softmax(pred_filtered[0].cpu().detach().numpy()) 
                        model3 = model3.eval()
                        pred_filtered = model2(torch.from_numpy(newBIM).reshape(1,3,224,224))
                        pred_filtered_BIM3 = softmax(pred_filtered[0].cpu().detach().numpy()) 

                    preds_ancestor_or_BIM.append(pred_filtered_BIM[class_dict[name][0]])
                    preds_ancestor_target_BIM.append(pred_filtered_BIM[class_dict[name][1]])
                    or_BIM_2.append(pred_filtered_BIM2[class_dict[name][0]])
                    target_BIM_2.append(pred_filtered_BIM2[class_dict[name][1]])
                    or_BIM_3.append(pred_filtered_BIM3[class_dict[name][0]])
                    target_BIM_3.append(pred_filtered_BIM3[class_dict[name][1]])
                    
                    maxBIM.append(pred_filtered_BIM.max())
                    maxBIM2.append(pred_filtered_BIM2.max())
                    maxBIM3.append(pred_filtered_BIM3.max())
                    
                    # EA: Get prediction of filtered image
                    with torch.no_grad():
                        model = model.eval()
                        pred_filtered = model(torch.from_numpy(newEA).reshape(1,3,224,224))
                        pred_filtered_EA = softmax(pred_filtered[0].cpu().detach().numpy())
                    with torch.no_grad():
                        model2 = model2.eval()
                        pred_filtered = model2(torch.from_numpy(newEA).reshape(1,3,224,224))
                        pred_filtered_EA2 = softmax(pred_filtered[0].cpu().detach().numpy()) 
                        model3 = model3.eval()
                        pred_filtered = model2(torch.from_numpy(newEA).reshape(1,3,224,224))
                        pred_filtered_EA3 = softmax(pred_filtered[0].cpu().detach().numpy()) 

                    preds_ancestor_or_EA.append(pred_filtered_EA[class_dict[name][0]])
                    preds_ancestor_target_EA.append(pred_filtered_EA[class_dict[name][1]])
                    or_EA_2.append(pred_filtered_EA2[class_dict[name][0]])
                    target_EA_2.append(pred_filtered_EA2[class_dict[name][1]])
                    or_EA_3.append(pred_filtered_EA3[class_dict[name][0]])
                    target_EA_3.append(pred_filtered_EA3[class_dict[name][1]])
                    
                    maxEA.append(pred_filtered_EA.max())
                    maxEA2.append(pred_filtered_EA2.max())
                    maxEA3.append(pred_filtered_EA3.max())

                # Plot probs vs. noise ratio
                f, ax = plt.subplots(3, 2, sharey=True, figsize=(10,5))

                ax[0,0].plot(np.log10(preds_ancestor_or_EA),label='$c_a$')
                ax[0,0].plot(np.log10(preds_ancestor_target_EA),label='$c_t$')
                ax[0,0].plot(np.log10(maxEA),label='max',color='r')

                ax[0,1].plot(np.log10(preds_ancestor_or_BIM))
                ax[0,1].plot(np.log10(preds_ancestor_target_BIM))
                ax[0,1].plot(np.log10(maxBIM),color='r')
                
                #2
                ax[1,0].plot(np.log10(or_EA_2))
                ax[1,0].plot(np.log10(target_EA_2))
                ax[1,0].plot(np.log10(maxEA2),color='r')

                ax[1,1].plot(np.log10(or_BIM_2))
                ax[1,1].plot(np.log10(target_BIM_2))
                ax[1,1].plot(np.log10(maxBIM2),color='r')

                #3
                ax[2,0].plot(np.log10(or_EA_3))
                ax[2,0].plot(np.log10(target_EA_3))
                ax[2,0].plot(np.log10(maxEA3),color='r')

                ax[2,1].plot(np.log10(or_BIM_3))
                ax[2,1].plot(np.log10(target_BIM_3))
                ax[2,1].plot(np.log10(maxBIM3),color='r')

                f.suptitle('Evolution of $o[a]$ and $o[t]$ with increasing noise magnitude',size=20,y=1.1)
                ax[1,0].set_title(r'ResNet152 EA',size=15)
                ax[1,1].set_title(r'ResNet152 BIM',size=15)
                ax[0,0].set_title(r'MobileNet EA',size=15)
                ax[0,1].set_title(r'MobileNet BIM',size=15)
                ax[2,0].set_title(r'DenseNet201 EA',size=15)
                ax[2,1].set_title(r'DenseNet201 BIM',size=15)
                
                
                ax[0,0].set_xlabel('f (%)',size=20)
                ax[0,1].set_xlabel('f (%)',size=20)
                ax[1,0].set_xlabel('f (%)',size=20)
                ax[1,1].set_xlabel('f (%)',size=20)
                ax[2,0].set_xlabel('f (%)',size=20)
                ax[2,1].set_xlabel('f (%)',size=20)
                
                ax[0,0].set_ylabel('Log probability',size=20)
                ax[0,1].set_ylabel('Log probability',size=20)
                ax[1,0].set_ylabel('Log probability',size=20)
                ax[1,1].set_ylabel('Log probability',size=20)
                ax[2,0].set_ylabel('Log probability',size=20)
                ax[2,1].set_ylabel('Log probability',size=20)
                
                f.text(0.5, 0.04, 'f', ha='center',fontsize=20)
                f.text(0.04, 0.5, 'Log probability', va='center', rotation='vertical',fontsize=20)                
                
                ax[0,0].set_xticks([])
                ax[0,1].set_xticks([])
                ax[1,0].set_xticks([])
                ax[1,1].set_xticks([])
                ax[2,0].set_xticks(range(0,65,5))
                ax[2,1].set_xticks(range(0,65,5))

                ax[2,0].set_xticklabels(list(range(0,325,25)),fontsize=15)
                ax[2,1].set_xticklabels(list(range(0,325,25)),fontsize=15)
                ax[2,0].xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax[2,1].xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax[2,0].tick_params(axis='y', which='major', labelsize=15)

                f.legend(prop={'size': 13}, loc='center right')

                file = analysis_results_path+"\\bandstop_graphs\\{}\\{}\\{}.png".format(attack_type,network,name)
                plt.savefig(file)
                f.subplots_adjust(hspace=0.4)
                plt.show()
