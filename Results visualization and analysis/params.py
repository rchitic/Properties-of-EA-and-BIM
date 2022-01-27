
#paths
data_path='C:\\Users\\raluca.chitic\\Desktop\\PhD\\tvs\\data'
results_path = 'C:\\Users\\raluca.chitic\\Desktop\\PhD\\tvs\\results'
analysis_results_path = 'C:\\Users\\raluca.chitic\\Desktop\\PhD\\tvs\\analysis\\results'

networks = ['VGG16','VGG19','ResNet50','ResNet101','ResNet152','DenseNet121','DenseNet169','DenseNet201','MobileNet','MNASNet']
class_dict = {'abacus':[398,641],'acorn':[988,947],'baseball':[429,541],'brown_bear':[294,150],'broom':[462,472],'canoe':[472,703],'hippopotamus':[344,368],'llama':[355,340],'maraca':[641,624],'mountain_bike':[671,807]}
names = list(class_dict.keys())
rgb = {0:'R',1:'G',2:'B'}