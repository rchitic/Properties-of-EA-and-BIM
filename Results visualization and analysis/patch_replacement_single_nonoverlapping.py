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
def show_graphs(patch_size):
    for network in networks:
        for name in names:
            print(f"{network} {name}:\n")

            path = results_path+'\\{}\\{}\\patch_replacement_nonoverlapping_single\\patch_size{}\\{}\\{}.npy'
            ancestorEA = np.load(path.format('EA',network,patch_size,name,'orig'))
            targetEA = np.load(path.format('EA',network,patch_size,name,'target'))
            ancestorBIM = np.load(path.format('BIM',network,patch_size,name,'orig'))
            targetBIM = np.load(path.format('BIM',network,patch_size,name,'target'))

            path_original_preds = results_path+"\\EA\\original_preds\\{}\\{}.npy".format(network,name)
            preds = np.load(path_original_preds)[0]
            ac, tc = class_dict[name][0], class_dict[name][1]
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            ax1.plot(np.log10(ancestorEA),label='$c_a$ EA')
            ax1.plot(np.log10(ancestorBIM),label='$c_a$ BIM')
            ax1.axhline(np.log10(preds[ac]), color="r")

            ax2.plot(np.log10(targetEA),label='$c_t$ EA')
            ax2.plot(np.log10(targetBIM),label='$c_t$ BIM')
            ax2.axhline(np.log10(preds[tc]), color="r")
            
            ax1.set_ylabel('log10(probability)')
            ax1.set_xlabel('replaced patch')
            ax2.set_xlabel('replaced patch')

            ax1.set_title('ancestor class')
            ax2.set_title('target class')
            f.suptitle("{} {} patch size {}".format(network,name,patch_size))
            
            ax1.legend(prop={'size': 13})
            ax2.legend(prop={'size': 13})
            
            plt.savefig(analysis_results_path+"\\nonoverlapping single patch replacement\\{}\\{}_patchSize{}.png".format(network,name,patch_size))
            plt.show()