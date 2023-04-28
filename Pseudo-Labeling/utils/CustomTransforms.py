import sys
sys.path.append('./')
print(sys.path)
import numpy as np
from torchvision import transforms
from fixmatch_randaugment import RandAugmentMC
# import MixMatch.dataset.cifar10 as mm_dataset



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


def get_fixmatch_transforms(dataset='cifar10'): # or 'uda', or 'mixmatch'
  

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


# def get_mixmatch_transforms(dataset='cifar10'):
#   means = (0.4914, 0.4822, 0.4465)
#   stds = (0.2471, 0.2435, 0.2616)
#   if dataset=='cifar10':
#     transform_train = transforms.Compose([
#           mm_dataset.RandomPadandCrop(32),
#           mm_dataset.RandomFlip(),
#           mm_dataset.ToTensor(),
#       ])
#     transform_unlabeled = mm_dataset.TransformTwice(transform_train)
#     transform_val = transforms.Compose([
#           mm_dataset.ToTensor(),
#     ])
  
#   return transform_train,transform_unlabeled,transform_val


# def get_pseudolabeling_transforms(dataset='cifar10'):
#     channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
#                          std=[0.2023, 0.1994, 0.2010])

#     if dataset == 'cifar10':
#         transform_train = transforms.Compose([
#             transforms.Pad(2, padding_mode='reflect'),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#             transforms.RandomCrop(32),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(**channel_stats)
#         ])

#         transform_unlabeled = mm_dataset.TransformTwice(transform_train)

#         transform_val = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(**channel_stats)
#         ])

#     return transform_train, transform_unlabeled, transform_val


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