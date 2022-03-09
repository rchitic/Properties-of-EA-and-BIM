networks = {1:'VGG16',2:'VGG19',3:'ResNet50',4:'ResNet101',5:'ResNet152',6:'DenseNet121',7:'DenseNet169',8:'DenseNet201',9:'MobileNet',10:'MNASNet',11:'BagNet17',12:'ResNet50_SIN'}
class_dict = {'abacus':[398,421],'acorn':[988,306],'baseball':[429,618],'brown_bear':[294,724],'broom':[462,273],'canoe':[472,176],'hippopotamus':[344,927],'llama':[355,42],'maraca':[641,112],'mountain_bike':[671,828]}
#targets = [bannister,rhinoceros beetle, ladle, dingo, pirate, saluki, trifle, agama, conch, strainer]
names = list(class_dict.keys())
alpha = 2/255
epsilon = 8/255

# EA
pop_size = 40
G = 103000

# BIM
N = 5

#data_path = '.../data/{}/{}{}.JPEG' (e.g. '.../data/{abacus/abacus1.JPEG')
#results_path = '...' (e.g. for 'results_path/EA/...' or 'results_path/BIM/...')
