import sys
BASE_DIR = '/content/drive/MyDrive/AdversarialAI/'
sys.path.insert(1, BASE_DIR + 'Poisoner')
from Poisoner import Poisoner
from get_datasets import get_subset_cifar10, get_poison_dataset_from_images
from CustomTransforms import get_fixmatch_transforms


if args.dataset == 'cifar10':
    args.num_classes = 10
    if args.arch == 'wideresnet':
        args.model_depth = 28
        args.model_width = 2
    elif args.arch == 'resnext':
        args.model_cardinality = 4
        args.model_depth = 28
        args.model_width = 4
train_labeled, train_unlabeled, test_dataset = get_subset_cifar10(subset_size=args.subset_size,
                                    labeled_size=args.labeled_size, 
                                    test_size=args.test_size,
                                    seed=args.seed)
transform_labeled, transform_unlabeled, transform_test = get_fixmatch_transforms(dataset='cifar10')

poison_dataset = poisoner.generate_poison_dataset(train_labeled, train_unlabeled, N=args.poison_size, 
                output_folder='/content/drive/MyDrive/Colab Notebooks/Tricks-of-Semi-supervisedDeepLeanring-Pytorch/')