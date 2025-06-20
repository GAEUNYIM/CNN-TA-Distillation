import torch
import torchvision
import torchvision.transforms as transforms

NUM_WORKERS = 2

def load_CIFAR(num_classes=100, dataset_dir='./data', batch_size=128, crop=False):

	normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
	simple_transform = transforms.Compose([transforms.ToTensor(), normalize])
	
	if crop is True:
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		])
	else:
		train_transform = simple_transform
	
	if num_classes == 100:
		trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
												 download=True, transform=train_transform)
		
		testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
												download=True, transform=simple_transform)
	else:
		trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
												 download=True, transform=train_transform)
		
		testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
												download=True, transform=simple_transform)
		
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS,
											  pin_memory=True, shuffle=True)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=NUM_WORKERS,
											 pin_memory=True, shuffle=False)
	return trainloader, testloader


if __name__ == "__main__":
	print("CIFAR10")
	print(get_cifar(10))
	print("---"*20)
	print("---"*20)
	print("CIFAR100")
	print(get_cifar(100))
