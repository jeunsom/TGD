from resnet import *
from mobilenet import *
from WideResNet import *
from vgg import *


from resnet_cifar import *
from plain_cnn_cifar import *


def is_resnet(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('resnet')

def is_wide_resnet(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('wide_resnet')

def is_vgg(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('vgg')

def is_mobile(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('mobilenet')

def is_wrn(name):
	"""
	Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
	:param name:
	:return:
	"""
	name = name.lower()
	return name.startswith('wrn')

def create_cnn_model(name, dataset="cifar100", use_cuda=False, num_cls=100, in_channel=3):
	"""
	Create a student for training, given student name and dataset
	:param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
	:param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
	:return: a pytorch student for neural network
	"""
	if dataset == 'cifar100':
		num_classes = 100
	elif dataset == 'gene':
		num_classes = num_cls #16
	elif dataset == 'pamap':
		num_classes = 12 
	else:
		num_classes = 10
	model = None
	if is_resnet(name) or is_wide_resnet(name):
		resnet_size = name[6:]
		if dataset == 'cifar100' or dataset == 'cifar10':
			resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes, in_channel=in_channel)
		else:
			print(dataset, 'dataset ', f'net: resnet-{resnet_size} cls: {num_classes}')
			#resnet_model = resnet_dict.get(resnet_size)(num_classes=num_classes) #res18 34
			resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes) #res cifar
		model = resnet_model
	elif is_wrn(name):
		wrn_size = name[3:]
		if dataset == 'cifar100' or dataset == 'cifar10':
			wrn_model = wrn_dict.get(wrn_size)(num_classes=num_classes, input_features=in_channel)
		else:
			print(dataset, 'dataset ', f'net: WResNet-{wrn_size} cls: {num_classes}')
			wrn_model = wrn_dict.get(wrn_size)(num_classes=num_classes, input_features=in_channel)
		model = wrn_model
	elif is_mobile(name):
		mobilenet_size = name[6:] 
		if dataset == 'cifar100' or dataset == 'cifar10':
			mobile_model = mobilenet_book.get(mobilenet_size)(num_classes=num_classes, in_channel=in_channel)
		else:
			print(dataset, 'dataset ', f'net: {mobilenet_size}net-v2 cls: {num_classes}')
			mobile_model = mobilenet_book.get(mobilenet_size)(num_classes=num_classes, in_channel=in_channel)
		model = mobile_model
	elif is_vgg(name):
		vgg_size = name[:]
		if dataset == 'cifar100' or dataset == 'cifar10':
			vgg_model = vgg_book.get(vgg_size)(num_classes=num_classes, in_channels=in_channel)
		else:
			print(dataset, 'dataset ', f'net: {vgg_size}net cls: {num_classes}')
			vgg_model = vgg_book.get(vgg_size)(num_classes=num_classes, in_channels=in_channel)
		model = vgg_model
	else:
		plane_size = name[5:]
		model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(plane_size)
		plane_model = ConvNetMaker(model_spec)
		model = plane_model

	# copy to cuda if activated
	if use_cuda:
		model = model.cuda()
		
	return model

# if __name__ == "__main__":
# 	dataset = 'cifar100'
# 	print('planes')
# 	for p in [2, 4, 6, 8, 10]:
# 		plane_name = "plane" + str(p)
# 		print(create_cnn_model(plane_name, dataset))
#
# 	print('-'*20)
# 	print("resnets")
# 	for r in [8, 14, 20, 26, 32, 44, 56, 110]:
# 		resnet_name = "resnet" + str(r)
# 		print(create_cnn_model(resnet_name, dataset))
