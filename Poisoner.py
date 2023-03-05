from scipy import stats
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms, datasets
import numpy as np
import imageio
from PIL import Image
from CustomTransforms import get_transforms
from get_datasets import get_subset_cifar10
import os
import shutil

class alpha_distribution(stats.rv_continuous):
    def _pdf(self, x):
        return 1.5 - x 


class Poisoner():
	def __init__(self):
		# for normalization
		self.means_stds = {
			'cifar10': {'mean': (0.4914, 0.4822, 0.4465), 'std':(0.2471, 0.2435, 0.2616)},
			'food101': {'mean': None, 'std': None},
			'inatural': {'mean': None, 'std': None},
		}
		self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

	def _get_random_alphas(self,size=10):
		distribution = alpha_distribution(a=0, b=1) # alpha in [0,1]
		alphas = distribution.rvs(size=size)
		return alphas

	# x_star is the sample we want to misclasify as some label y_target,
	# x_target is a random sample with that label y_target
	def _get_adv_samples(self, x_star, x_target, size):
		alphas = self._get_random_alphas(size)
		X_adv = []
		for a in alphas:
			interp = (1-a)*x_target + a*x_star
			interp = np.rint(interp).astype(np.uint8)
			X_adv.append(interp)
		X_adv = np.array(X_adv)
		return X_adv

	def _get_img_from_torch(self, img):
		img = img.T*255
		img = np.swapaxes(img, 0, 1)
		img = img.astype(np.uint8)
		return img


	def _store_chosen_samples(self, path='./data/poison/', name='latest_poison', malicious_sample=None, target_sample=None):
		assert self.X_adv is not None

		folder_path = path + name
		if os.path.isdir(folder_path):
			shutil.rmtree(folder_path)
		os.mkdir(folder_path)

		x_mali, y_mali = malicious_sample
		x_target, y_target = target_sample

		# mali_img = self._get_img_from_torch(x_mali)
		# target_img = self._get_img_from_torch(x_target)

		mali_name = self.label_names[y_mali]
		target_name = self.label_names[y_target]

		imageio.imsave(folder_path + "/source_{}.jpg".format(target_name), x_target)
		imageio.imsave(folder_path + "/mali_{}.jpg".format(mali_name), x_mali)
		print("target and malicious images saved to", folder_path)
		

	def generate_poison_loader(self, dataset, N):
		loader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=True)
		loader = iter(loader)
		
		imgs, labels = next(loader)
		imgs, labels = imgs.numpy(), labels.numpy()
		x_star, y_star = imgs[0], labels[0] # random x_star, already shuffled earlier

		all_labels = list(range(10))
		all_labels.remove(y_star) # remove the true label of x_star
		y_target = np.random.choice(all_labels) # something different than y_star

		print("malicious:", y_star, ", target:", y_target)

		# find random x_target with the label y_target
		x_target = None
		for imgs, labels in loader:
			imgs, labels = imgs.numpy(), labels.numpy()
			indices_with_y_target = np.where(labels==y_target)[0]
			if len(indices_with_y_target) == 0:
				continue
			else: # take the first case with y_target, since already shuffled earlier
				idx = np.random.choice(indices_with_y_target)
				x_target = imgs[idx]

		# get interpolated samples to inject
		self.X_adv = self._get_adv_samples(x_star, x_target, N)
		self.y_adv = y_star

		# generate dataset
		poison_dataset = CIFAR_POISON(data=self.X_adv, label=y_star, transform=None)

		# save for evaluation
		malicious_sample = (x_star, y_star)
		target_sample = (x_target, y_target)
		self._store_chosen_samples(name='latest_poison', malicious_sample=malicious_sample, target_sample=target_sample)

		return poison_dataset



class CIFAR_POISON(Dataset):
	def __init__(self, data, label, transform=None, asPIL=False):
		self.data = data
		self.label = label
		self.transform = transform
		self.asPIL = asPIL

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img = self.data[idx]
		label = self.label # unlabeled, but keep for evaluation purposes

		if self.asPIL:
			img = Image.fromarray(img)

		if self.transform:
			img = self.transform(img)

		return img, label

def main():
	train_labeled, train_unlabeled, test_dataset = get_subset_cifar10()
	transform_labeled, transform_unlabeled, transform_test = get_transforms(method='fixmatch', dataset='cifar10')

	poisoner = Poisoner()
	poison_dataset = poisoner.generate_poison_loader(train_unlabeled, 10)
	poison_dataset.transform = transform_unlabeled
	poison_dataset.asPIL = True

	train_unlabeled.transform = transform_unlabeled
	train_unlabeled.asPIL = True

	print(poison_dataset)
	print(len(poison_dataset))

	# loader = DataLoader(poison_dataset, batch_size=2, num_workers=1)
	combined = ConcatDataset([train_unlabeled, poison_dataset])
	loader = DataLoader(combined, batch_size=2, num_workers=1, shuffle=True)

	i = 0
	for (w,s),labels in loader:
		print(labels, labels.numpy())
		print(type(labels), type(w))
		break		



if __name__ == '__main__':
	main()