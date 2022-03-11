import numpy as np
import matplotlib.pyplot as plt
import params

# params
#-------------------------------------------------------------------------------------------------------------------------------------------------
networks = params.networks
class_dict = params.class_dict
names = params.names
results_path = params.results_path
analysis_results_path = params.analysis_results_path

#-------------------------------------------------------------------------------------------------------------------------------------------------
def show_heatmap(attack_type):
    for network in networks:
        for name in names:
            for order in range(1,11):
                # Get results
                # adversarial
                path_bagnet9 = results_path+'\\{}\\{}\\heatmap\\BagNet{}\\{}\\{}{}.npy'
                path_bagnet17 = results_path+'\\{}\\{}\\heatmap\\BagNet{}\\{}\\{}{}.npy'
                path_bagnet33 = results_path+'\\{}\\{}\\heatmap\\BagNet{}\\{}\\{}{}.npy'
                print(f"{network} {name}:\n")
                bagnet9_or = np.load(path_bagnet9.format(attack_type,network,'9',name,'original',order))
                bagnet17_or = np.load(path_bagnet17.format(attack_type,network,'17',name,'original',order))
                bagnet33_or = np.load(path_bagnet33.format(attack_type,network,'33',name,'original',order))
                bagnet9_target = np.load(path_bagnet9.format(attack_type,network,'9',name,'target',order))
                bagnet17_target = np.load(path_bagnet17.format(attack_type,network,'17',name,'target',order))
                bagnet33_target = np.load(path_bagnet33.format(attack_type,network,'33',name,'target',order))
                or_, target = class_dict[name][0], class_dict[name][1]
                f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, figsize=(13,10))

               # ancestor
                ancestor_or = np.load(results_path+'\\EA\\heatmap_BagNet17\\{}\\original{}.npy'.format(name,order))
                ancestor_target = np.load(results_path+'\\EA\\heatmap_BagNet17\\{}\\target{}.npy'.format(name,order))

                # Display heatmaps
                # adversarial
                im1 = ax1.imshow(bagnet17_or.reshape(224,224) > np.median(bagnet33_or),label='adv ac')
                f.colorbar(im1,ax=ax1)
                im2 = ax2.imshow(bagnet17_target.reshape(224,224) > np.median(bagnet33_target),label='adv tc')
                f.colorbar(im2,ax=ax2)
                ax1.set_title('adv ac')
                ax2.set_title('adv tc')
            
                # ancestor
                im1 = ax3.imshow(ancestor_or.reshape(224,224) > np.median(ancestor_or),label='ancestor ac')
                f.colorbar(im1,ax=ax3)
                im2 = ax4.imshow(ancestor_target.reshape(224,224) > np.median(ancestor_target),label='ancestor tc')
                f.colorbar(im2,ax=ax4)
                ax3.set_title('ancestor ac')
                ax4.set_title('ancestor tc')
            
                # diff
                im1 = ax5.imshow(bagnet17_or.reshape(224,224) - ancestor_or.reshape(224,224) > 0,label='ancestor ac')
                f.colorbar(im1,ax=ax5)
                im2 = ax6.imshow(bagnet17_target.reshape(224,224) - ancestor_target.reshape(224,224) > 0,label='ancestor tc')
                f.colorbar(im2,ax=ax6)
                ax5.set_title('diff ac')
                ax6.set_title('diff tc')
            
                f.suptitle("{} attack, {} {}".format(attack_type,network,name,order))
                plt.savefig(analysis_results_path+"\\heatmap\\{}\\{}\\{}{}.png".format(attack_type,network,name,order))
                plt.show()

def show_heatmap_quantiles(attack_type,network,name,patch_size):

    #texture of ancestors
    ancestor_ca = np.load(results_path+'\\EA\\heatmap_BagNet17\\{}\\original{}.npy'.format(name,order))
    ancestor_ct = np.load(results_path+'\\EA\\heatmap_BagNet17\\{}\\target{}.npy'.format(name,order))
    #texture of CNN adversarials
    path_bagnet17 = results_path+'\\{}\\{}\\heatmap\\BagNet{}\\{}\\{}{}.npy'
    BagNeteffect_ca = np.load(path_bagnet17.format(attack_type,network,'17',name,'original',order))
    BagNeteffect_ct = np.load(path_bagnet17.format(attack_type,network,'17',name,'target',order))

    path = results_path+'\\{}\\{}\\patch_replacement_overlapping\\patch_size{}\\{}\\{}{}.npy'
    CNNeffect_ca = np.load(path.format(attack_type,network,patch_size,name,'orig',order))
    CNNeffect_ct = np.load(path.format(attack_type,network,patch_size,name,'target',order))

    f, axarr = plt.subplots(1,5,figsize=(20,15))
    axarr[0].imshow(((BagNeteffect_ca_EA - ancestor_ca)<np.quantile(BagNeteffect_ca_EA - ancestor_ca,0.1)).reshape(224,224),interpolation='none')
    axarr[1].imshow(((BagNeteffect_ct_EA - ancestor_ct)>np.quantile(BagNeteffect_ct_EA - ancestor_ct,0.9)).reshape(224,224),interpolation='none')
    axarr[2].imshow(((BagNeteffect_ca_EA - BagNeteffect_ct_EA)>np.quantile(BagNeteffect_ca_EA - BagNeteffect_ct_EA,0.9)).reshape(224,224),interpolation='none')
    axarr[3].imshow(((BagNeteffect_ct_EA - BagNeteffect_ca_EA)>np.quantile(BagNeteffect_ct_EA - BagNeteffect_ca_EA,0.9)).reshape(224,224),interpolation='none')
    axarr[4].imshow((CNNeffect_ct_EA > np.quantile(CNNeffect_ct_EA,0.9)).reshape(224,224),interpolation='none')
    for i in range(4):
        axarr[i].axis('off')
    axarr[0].set_title('diff texture o[a]',fontsize=20)
    axarr[1].set_title('diff texture o[t]',fontsize=20)
    axarr[2].set_title('texture o[a]>o[t]',fontsize=20)
    axarr[3].set_title('texture o[t]>o[a]',fontsize=20)
    axarr[4].set_title('diff CNN o[t]',fontsize=20)
