from scipy import stats
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms, datasets
import numpy as np
import imageio
from PIL import Image

from CustomTransforms import get_transforms
from get_datasets import get_subset_cifar10, CIFAR_POISON
import os
import shutil

import pdb

class alpha_distribution(stats.rv_continuous):
    def _pdf(self, x):
        return 1.5 - x 


class Poisoner():
  def __init__(self, dataset="cifar10"):
    self.dataset = dataset
    self.cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    self.fashion_mnist_label_names = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    self.mnist_label_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


    if dataset=="cifar10":
      self.label_names = self.cifar10_label_names
    elif dataset =="mnist":
      self.label_names = self.mnist_label_names
    elif dataset =="fashion_mnist":
      self.label_names = self.fashion_mnist_label_names


  def _get_random_alphas(self,size=10):
    distribution = alpha_distribution(a=0, b=1) # alpha in [0,1]
    alphas = distribution.rvs(size=size)
    return alphas

  # x_star is the sample we want to misclasify as some label y_target,
  # x_target is a random sample with that label y_target
  def _get_adv_samples(self, x_star, x_target, size):
    self.alphas = sorted(self._get_random_alphas(size))
    X_adv = []
    for a in self.alphas:
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

  def _read_stored_samples(self,input_folder,target_sample= None):
    x_mali_list = []
    y_mali_list = []
    x_target_list = []
    y_target_list = []
    injected_list = []

    target_prefix = "source_"
    mali_prefix = "mali_"
    injected_prefix = "injected_"

    for filename in os.listdir(input_folder+"/injected"):
        injected_img = imageio.imread(os.path.join(input_folder+"/injected", filename))
        if injected_img.ndim ==2:
          injected_img = np.expand_dims(injected_img, axis=0)
        alpha = float(filename[len(injected_prefix):-4])  # remove prefix and file extension
        injected_list.append(injected_img)

    self.X_adv = np.array(injected_list)
    
   
    self.y_adv = target_sample
    poison_dataset = CIFAR_POISON(data=self.X_adv, label=self.y_adv, transform=None)
    return poison_dataset

  def _store_chosen_samples(self, output_folder=None, malicious_sample=None, target_sample=None):
    assert self.X_adv is not None
    folder_path = output_folder
    if os.path.isdir(folder_path):
      shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    x_mali, y_mali = malicious_sample
    x_target, y_target = target_sample

    mali_name = self.label_names[y_mali]
    target_name = self.label_names[y_target]
    if x_target.shape[0]==1:
      x_target = x_target.squeeze(0)
      x_mali = x_mali.squeeze(0)
    # pdb.set_trace()
    imageio.imsave(folder_path + "/source_{}.jpg".format(target_name), x_target)
    imageio.imsave(folder_path + "/mali_{}.jpg".format(mali_name), x_mali)

    # save injected poison
    injected_folder = folder_path + '/injected'
    os.mkdir(injected_folder)
    for i in range(len(self.X_adv)):
      injected_img = self.X_adv[i]
      if injected_img.shape[0]==1:
        injected_img = injected_img.squeeze(0)
      alpha = np.round(self.alphas[i], decimals=3)
      imageio.imsave(injected_folder + '/injected_{}.jpg'.format(alpha), injected_img)
    
    
    print("target and malicious images saved to", folder_path)
    

  def generate_poison_dataset(self, labeled_dataset, unlabeled_dataset, N, output_folder='./data/poison/latest_poison', label_pair=None, dataset = "cifar10"):
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, num_workers=1, shuffle=True)
    labeled_loader = iter(labeled_loader)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, num_workers=1, shuffle=True)
    unlabeled_loader = iter(unlabeled_loader)

    if label_pair is None:
      # get random labels for x_star and x_target
      all_labels = list(range(10))
      y_star, y_target = np.random.choice(all_labels, size=2, replace=False)
    else:
      # specified labels
      y_star, y_target = label_pair # [malicious, source]

    print("malicious:", self.label_names[y_star], ", target:", self.label_names[y_target])

    # find random x_star with the label y_star
    x_star = None
    for imgs, labels in unlabeled_loader:
      imgs, labels = imgs.numpy(), labels.numpy()
      indices_with_y_target = np.where(labels==y_star)[0]
      if len(indices_with_y_target) == 0:
        continue
      else: # take the first case with y_target, since already shuffled earlier
        idx = np.random.choice(indices_with_y_target)
        x_star = imgs[idx]
    print("x_star:", type(x_star))

    # find random x_target with the label y_target
    x_target = None
    for imgs, labels in labeled_loader:
      imgs, labels = imgs.numpy(), labels.numpy()
      indices_with_y_target = np.where(labels==y_target)[0]
      if len(indices_with_y_target) == 0:
        continue
      else: # take the first case with y_target, since already shuffled earlier
        idx = np.random.choice(indices_with_y_target)
        x_target = imgs[idx]
    print("x_target:", type(x_target))

    # get interpolated samples to inject
    self.X_adv = self._get_adv_samples(x_star, x_target, N)
    self.y_adv = y_star

    # generate dataset
    poison_dataset = CIFAR_POISON(data=self.X_adv, label=y_star, transform=None)

    # save for evaluation
    malicious_sample = (x_star, y_star)
    target_sample = (x_target, y_target)
    self._store_chosen_samples(output_folder=output_folder, malicious_sample=malicious_sample, target_sample=target_sample)

    return poison_dataset

	


def main():
	train_labeled, train_unlabeled, test_dataset = get_subset_cifar10()
	transform_labeled, transform_unlabeled, transform_test = get_transforms(method='fixmatch', dataset='cifar10')

	poisoner = Poisoner()
	poison_dataset = poisoner.generate_poison_dataset(train_labeled, train_unlabeled, 10)
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