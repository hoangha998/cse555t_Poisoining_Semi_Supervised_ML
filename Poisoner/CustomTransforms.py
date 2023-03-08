import numpy as np
from torchvision import transforms
from fixmatch_randaugment import RandAugmentMC

# Contains all transforms needed for unlabeled dataset
#    - method: fixmatch, uda, or mixmatch
#    - dataset: cifar10, food101, or inatural
def get_transforms(method='fixmatch', dataset='cifar10'):
  if method=='fixmatch':
    return get_fixmatch_transforms(dataset)
  elif method=='mixmatch':
    return get_mixmatch_transforms(dataset) # To be implemented
  elif method=='uda': 
    return get_uda_transforms(dataset) # To be implemented


def get_fixmatch_transforms(dataset='cifar10'): # or 'uda', or 'mixmatch'
  cifar10_mean = (0.4914, 0.4822, 0.4465)
  cifar10_std = (0.2471, 0.2435, 0.2616)

  if dataset=='cifar10':
      transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                    padding=int(32*0.125),
                    padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
      ])

      transform_unlabeled = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
      
      transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
      ])

      return transform_labeled, transform_unlabeled, transform_test


def get_mixmatch_transforms(dataset='cifar10'):
  pass # TODO


def get_uda_transforms(dataset='cifar10'):
  pass # TODO
  

class TransformFixMatch(object):
    def __init__(self, mean, std):
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
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)