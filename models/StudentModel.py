from ModelResNet import *
from ModelCNN import *

def check_model(name):
	name = name.lower()
	return name.startswith('resnet')

def create_student_model(name, dataset="cifar100", use_cuda=False):
	# Return student model for training
	num_classes = 100 if dataset == 'cifar100' else 10
	model = None
	if check_model(name):
		resnet_size = name[6:]
		resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes)
		model = resnet_model
	else:
		plane_size = name[5:]
		model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(plane_size)
		plane_model = ConvNetMaker(model_spec)
		model = plane_model
	if use_cuda:
		model = model.cuda()	
	return model

