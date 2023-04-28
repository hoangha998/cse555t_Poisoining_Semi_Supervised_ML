import sys
sys.path.append('./')
print(sys.path)
import numpy as np
from torchvision import transforms
from fixmatch_randaugment import RandAugmentMC
import mixmatch_augment as mm_dataset
import torch



# Contains all transforms needed for unlabeled dataset
#    - method: fixmatch, pesudolabeling, or mixmatch

#    - dataset: cifar10, food101, or inatural
def get_transforms(method='fixmatch', dataset='cifar10'):
  if method=='fixmatch':
    return get_fixmatch_transforms(dataset)
  elif method=='mixmatch':
    return get_mixmatch_transforms(dataset) # To be implemented
  elif method=='pseudolabeling':
    return get_pseudolabeling_transforms(dataset) # To be implemented
  elif method=='uda': 
    return get_uda_transforms(dataset) # To be implemented



def get_fixmatch_transforms(dataset='cifar10'): # or 'uda', or 'mixmatch'
  cifar10_mean = (0.4914, 0.4822, 0.4465)
  cifar10_std = (0.2471, 0.2435, 0.2616)

  if dataset=='cifar10':
      print("getting transform for cifar10")
      transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                    padding=int(32*0.125),
                    padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
      ])

      transform_unlabeled = TransformFixMatch(mean=cifar10_mean, std=cifar10_std, mnist=False)
      
      transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
      ])

      return transform_labeled, transform_unlabeled, transform_test

  elif dataset=='mnist' or dataset=='fashion_mnist':
      print("getting transform for", dataset)
      mnist_mean = (0.28604, 0.28604, 0.28604)
      mnist_std = (0.35302, 0.35302, 0.35302)
      if dataset == 'mnist':
          mnist_mean = (0.13066, 0.13066, 0.13066)
          mnist_std = (0.30811, 0.30811, 0.30811)

      transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                    padding=int(28*0.125),
                    padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Lambda(replicate_channels),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
      ])

      transform_unlabeled = TransformFixMatch(mean=mnist_mean, std=mnist_std, mnist=True)
      
      transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(replicate_channels),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
      ])

      return transform_labeled, transform_unlabeled, transform_test

      




def replicate_channels(img):
    return torch.cat([img, img, img], dim=0)

def get_mixmatch_transforms(dataset='cifar10'):
  if dataset == 'cifar10':
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2471, 0.2435, 0.2616)
    transform_train = transforms.Compose([
          mm_dataset.Normalize(),
          mm_dataset.RandomPadandCrop(32),
          mm_dataset.RandomFlip(),
          mm_dataset.ToTensor(),
      ])
    transform_unlabeled = mm_dataset.TransformTwice(transform_train)
    transform_val = transforms.Compose([
          mm_dataset.Normalize(),
          mm_dataset.ToTensor(),
    ])
  elif dataset == "fashion_mnist" or dataset == "mnist":
      means = (0.4914)
      stds = (0.2471)
      transform_train = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),  
            mm_dataset.RandomPadandCrop(32),
            mm_dataset.RandomFlip(),
            mm_dataset.ToTensor(),
            transforms.Lambda(replicate_channels),
            transforms.Normalize(
                (0.5,0.5,0.5), (0.5,0.5,0.5)
            ),  # Normalize the pixel values using the mean and standard deviation of Fashion MNIST
        ])
      transform_unlabeled = transforms.Compose([
            # mm_dataset.ToTensor(),
            # transforms.Grayscale(num_output_channels=1),  
            mm_dataset.RandomPadandCrop(32),
            mm_dataset.RandomFlip(),
            mm_dataset.ToTensor(),
            transforms.Lambda(replicate_channels),
            transforms.Normalize(
                (0.5,0.5,0.5), (0.5,0.5,0.5)
            ),  # Normalize the pixel values using the mean and standard deviation of Fashion MNIST
        ])
      transform_unlabeled = mm_dataset.TransformTwice(transform_unlabeled)

      transform_val = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),  
            mm_dataset.RandomPadandCrop(32),
            mm_dataset.ToTensor(),
            transforms.Lambda(replicate_channels),
            transforms.Normalize(
                (0.5,0.5,0.5), (0.5,0.5,0.5)
            ),  # Normalize the pixel values using the mean and standard deviation of Fashion MNIST
            
      ])
  
  return transform_train,transform_unlabeled,transform_val


def get_pseudolabeling_transforms(n_labels, dataset='cifar10'):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                std = [0.2023, 0.1994, 0.2010])
    if dataset == 'cifar10':
      train_transform = transforms.Compose([
          transforms.Pad(2, padding_mode='reflect'),
          transforms.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.4, hue=0.1),
          transforms.RandomCrop(32),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(**channel_stats)
      ])
      eval_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(**channel_stats)
      ])
      trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                    transform=train_transform)
      evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                    transform=eval_transform)
      num_classes = 10
      label_per_class = n_labels // num_classes
      labeled_idxs, unlabed_idxs = split_relabel_data(
                        np.array(trainset.targets),
                        trainset.targets,
                        label_per_class,
                        num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }


def get_uda_transforms(dataset='cifar10'):
  pass # TODO
  


class TransformFixMatch(object):
    def __init__(self, mean, std, mnist=False):
        if mnist:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=28,
                                    padding=int(28*0.125),
                                    padding_mode='reflect')
                                    ]
                )
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=28,
                                    padding=int(28*0.125),
                                    padding_mode='reflect'
                                    ),
                RandAugmentMC(n=2, m=10, dim=1)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(replicate_channels),
                transforms.Normalize(mean=mean, std=std)])
        else:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                    padding=int(32*0.125),
                                    padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                    padding=int(32*0.125),
                                    padding_mode='reflect'),
                RandAugmentMC(n=2, m=10, dim=3)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

        

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)