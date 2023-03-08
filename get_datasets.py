from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from CustomTransforms import get_transforms
import imageio

# def get_cifar10(num_samples=10000, num_classes=10, train=True):
# 	dataset = datasets.CIFAR10('./data', train=train, transform=transforms.ToTensor(), download=True, shuffle=True)
# 	label_names = ['airplane', 'automobile']
# 	loader = DataLoader(
# 		dataset,
# 		batch_size=16,
# 		num_workers=1,
# 		drop_last=True)

# 	num_samples_per_class = num_samples//num_classes
# 	classes_counts = [0]*num_classes

# 	i = 0
# 	for images, label in loader:
# 		img = images[12].numpy()
# 		test_img = img.T*255
# 		test_img = np.swapaxes(test_img, 0, 1)
# 		test_img = test_img.astype(np.uint8)
# 		print(label.numpy()[12])
# 		# print(img)
# 		# print(type(img))
# 		imageio.imsave('test.jpg', test_img)
# 		i+=1
# 		if i > 0: 
# 			break

# Return an object of class torchvision.datasets 
def get_subset_cifar10(num_classes=10, subset_size=10000, labeled_ratio=0.05, 
					  test_size=2000, transform_labeled=None, transform_unlabeled=None, 
					  seed=99):
	base_dataset = datasets.CIFAR10('./data', train=True, download=True)
	test_dataset = datasets.CIFAR10('./data', train=False, download=True)

	# --- Training data ---
	train_all_labels = np.array(base_dataset.targets)
	samples_per_class = subset_size // num_classes
	labled_per_class = int(samples_per_class * labeled_ratio)

	labeled_indices = []
	unlabeled_indices = []
	for i in range(num_classes):
		# get a subset of the originial using indices
		class_indices = np.where(train_all_labels == i)[0]
		class_indices = np.random.choice(class_indices, samples_per_class, False)
		np.random.shuffle(class_indices) # shuffle

		# split that subset to labeled and unlabled 
		labeled_indices.extend(class_indices[:labled_per_class])
		unlabeled_indices.extend(class_indices[labled_per_class:])

	train_labeled_indices = np.array(labeled_indices)
	train_unlabeled_indices = np.array(unlabeled_indices)

	# --- Testing data ---
	test_all_labels = np.array(test_dataset.targets) 
	test_per_class = test_size // num_classes
	test_indices = []
	for i in range(num_classes):
		# get a subset of the originial using indices
		class_indices = np.where(test_all_labels == i)[0]
		class_indices = np.random.choice(class_indices, test_per_class, False)
		np.random.shuffle(class_indices) # shuffle
		test_indices.extend(class_indices)

	test_indices = np.array(test_indices)

	# --- Create dataset objects and return ---
	np.random.shuffle(train_labeled_indices)
	np.random.shuffle(train_unlabeled_indices)
	np.random.shuffle(test_indices)

	train_labeled = CIFAR10_SSL(train_labeled_indices, train=True, transform=None, 
									download=True)
	train_unlabeled = CIFAR10_SSL(train_unlabeled_indices, train=True, transform=None, 
									download=True)
	test_dataset = CIFAR10_SSL(test_indices, train=False, transform=None, 
									download=True)

	return train_labeled, train_unlabeled, test_dataset


# Custom dataset using a subset of the original specified by indices
class CIFAR10_SSL(datasets.CIFAR10):
	def __init__(self, indices, train=True,
				transform=None, target_transform=None,
				download=False, asPIL=False):
		super().__init__('./data', train=train,
		transform=transform,
		target_transform=target_transform,
		download=download)

		self.asPIL = asPIL
		if indices is not None:
			self.data = self.data[indices]
			self.targets = np.array(self.targets)[indices]


	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		if self.asPIL:
			img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	

if __name__ == '__main__':
	train_labeled, train_unlabeled, test_dataset = get_subset_cifar10()
	# transform_labeled, transform_unlabeled, transform_test = get_transforms(method='fixmatch', dataset='cifar10')
	# train_labeled.transform = transform_labeled
	# print(train_labeled)
	# print(train_unlabeled)
	# print(test_dataset)
	# combined = ConcatDataset([train_labeled, train_unlabeled])
	combined_loader = DataLoader(train_labeled, batch_size=2, num_workers=1)
	i = 0
	combined_loader = iter(combined_loader)
	print("getting a sample")
	a,b = next(combined_loader)
	print("a:", type(a))
	img = a.numpy()[0]
	label = b.numpy()[0]
	print(img)
	print(img.shape, type(img), type(img[0,0,0]))
	# print(b.numpy())
	# print("combined data has", i, "samples")

	# base_dataset = datasets.CIFAR10('./data', train=True, download=True)
	# base_dataset = base_dataset[[0,1,2]]
	# print(len(base_dataset))
	# print(base_dataset[0])