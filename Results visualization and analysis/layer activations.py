import numpy as np
import matplotlib.pyplot as plt
import pickle
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

locations = {-1:0,0:2,1:4}

#-------------------------------------------------------------------------------------------------------------------------------------------------
def plot_activations(quartile,oom):
    '''
    quartile - string, one of ['1','2','3','4']
    oom - int, one of [-1,0,1]
    '''

    path = results_path+'\\{}\\activations_ancestors_stats_quartile{}_complete{}.pickle'
    EA = pickle.load(open(path.format('EA',quartile,'EA'), "rb"))
    BIM = pickle.load(open(path.format('BIM',quartile,'BIM'), "rb"))
    for name in names:
        for network in networks:
            loc_1 = locations[oom]
            loc_2 = locations[oom] + 1

            EA_1 = [EA[name][network][layer][oom][0] for layer in range(len(EA[name][network]))]
            EA_2 = [EA[name][network][layer][oom][1] for layer in range(len(EA[name][network]))]
            EA_3 = [EA[name][network][layer][oom][2] for layer in range(len(EA[name][network]))]
            EA_4 = [EA[name][network][layer][oom][3] for layer in range(len(EA[name][network]))]
            
            BIM_1 = [BIM[name][network][layer][oom][0] for layer in range(len(BIM[name][network]))]
            BIM_2 = [BIM[name][network][layer][oom][1] for layer in range(len(BIM[name][network]))]
            BIM_3 = [BIM[name][network][layer][oom][2] for layer in range(len(BIM[name][network]))]
            BIM_4 = [BIM[name][network][layer][oom][3] for layer in range(len(BIM[name][network]))]
            
            description = '% where log10 ((A(adv)-A(ancestor)) / A(ancestor)) = {}'.format(oom)
            labels = ['q1_negInit_negChange','q1_negInit_posChange','q1_posInit_negChange','q1_posInit_posChange']

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
            f.suptitle("{} {}: {}".format(name,network,description),size=16)
            
            EA_1 = np.array(EA_1)
            EA_2 = np.array(EA_2)
            EA_3 = np.array(EA_3)
            EA_4 = np.array(EA_4)
            BIM_1 = np.array(BIM_1)
            BIM_2 = np.array(BIM_2)
            BIM_3 = np.array(BIM_3)
            BIM_4 = np.array(BIM_4)
            limitmin = np.min([np.min(EA_1[~np.isnan(EA_1)]),np.min(EA_2[~np.isnan(EA_2)]),np.min(EA_3[~np.isnan(EA_3)]),np.min(EA_4[~np.isnan(EA_4)]),np.min(BIM_1[~np.isnan(BIM_1)]),np.min(BIM_2[~np.isnan(BIM_2)]),np.min(BIM_3[~np.isnan(BIM_3)]),np.min(BIM_4[~np.isnan(BIM_4)])])
            limitmax = np.max([np.max(EA_1[~np.isnan(EA_1)]),np.max(EA_2[~np.isnan(EA_2)]),np.max(EA_3[~np.isnan(EA_3)]),np.max(EA_4[~np.isnan(EA_4)]),np.max(BIM_1[~np.isnan(BIM_1)]),np.max(BIM_2[~np.isnan(BIM_2)]),np.max(BIM_3[~np.isnan(BIM_3)]),np.max(BIM_4[~np.isnan(BIM_4)])])

            ax1.plot(EA_1,label=labels[0])
            ax1.plot(EA_2,label=labels[1])
            ax1.plot(EA_3,label=labels[2])
            ax1.plot(EA_4,label=labels[3])
            ax1.set_title("EA")
            ax1.set_ylim(limitmin,limitmax)
            if type(oom) != int:
                ax1.set_ylabel('%',fontsize=14)
                ax1.axhline(0, color="r")
            ax1.legend()
            ax1.set_xlabel('Conv layer',fontsize=14)

            ax2.plot(BIM_1,label=labels[0])
            ax2.plot(BIM_2,label=labels[1])
            ax2.plot(BIM_3,label=labels[2])
            ax2.plot(BIM_4,label=labels[3])
            ax2.set_title("BIM")
            ax2.set_ylim(limitmin,limitmax)
            if type(oom) != int:
                ax2.set_ylabel('%',fontsize=14)
                ax2.axhline(0, color="r")
            ax2.legend()
            ax2.set_xlabel('Conv layer',fontsize=14)
            
            plt.show()