
from re import split

from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from CustomTransforms import get_transforms

import pdb

# import CustomTransforms.get_transforms
import imageio
import os

DATA_PATH = "./data"
import PIL.Image

def get_subset_mnist(
    num_classes=10,
    subset_size=20000,
    labeled_size=200,
    test_size=4000,
    transform_labeled=None,
    transform_unlabeled=None,
    seed=99,
    with_val=False,
    image_size=(28, 28),
    unsqueeze=True
):
    np.random.seed(seed)
    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize the image to the specified size
            transforms.ToTensor(),  # Convert the image to a tensor with pixel values in the range [0,1]
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize the pixel values using the mean and standard deviation of MNIST
        ]
    )

    # Load the datasets
    print(
        "Getting MNIST | Subset size, labeled size, test size =",
        [subset_size, labeled_size, test_size],
    )
    base_dataset = datasets.MNIST(
        DATA_PATH, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        DATA_PATH, train=False, download=True, transform=transform
    )

    # --- Training data ---
    train_all_labels = np.array(base_dataset.targets)
    samples_per_class = subset_size // num_classes
    labeled_per_class = labeled_size // num_classes

    labeled_indices = []
    unlabeled_indices = []
    for i in range(num_classes):
        # get a subset of the original using indices
        class_indices = np.where(train_all_labels == i)[0]
        class_indices = np.random.choice(class_indices, samples_per_class, False)
        np.random.shuffle(class_indices)  # shuffle

        # split that subset to labeled and unlabeled
        labeled_indices.extend(class_indices[:labeled_per_class])
        unlabeled_indices.extend(class_indices[labeled_per_class:])

    train_labeled_indices = np.array(labeled_indices)
    train_unlabeled_indices = np.array(unlabeled_indices)

    # --- Testing data ---
    test_all_labels = np.array(test_dataset.targets)
    test_per_class = test_size // num_classes
    test_indices = []
    for i in range(num_classes):
        # get a subset of the original using indices
        class_indices = np.where(test_all_labels == i)[0]
        class_indices = np.random.choice(class_indices, test_per_class, False)
        np.random.shuffle(class_indices)  # shuffle
        test_indices.extend(class_indices)

    test_indices = np.array(test_indices)

    # --- Create dataset objects and return ---
    np.random.shuffle(train_labeled_indices)
    np.random.shuffle(train_unlabeled_indices)
    np.random.shuffle(test_indices)

    val_indices = test_indices[
        0 : int(test_size * 0.1)
    ]  # validation set is 10% of the test set

    test_indices = test_indices[int(test_size * 0.1) :]

    train_labeled = MNIST_SSL(
        indices=train_labeled_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=True,
        unsqueeze=unsqueeze
    )
    train_unlabeled = MNIST_SSL(
        indices=train_unlabeled_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=True,
        unsqueeze=unsqueeze
    )
    test_dataset = MNIST_SSL(
        indices=test_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=False,
        unsqueeze=unsqueeze
    )

    val_dataset = MNIST_SSL(
        indices=val_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=False,
        unsqueeze=unsqueeze
    )
    print(
        "labeled, unlabeled, test sizes:",
        [len(train_labeled), len(train_unlabeled), len(test_dataset)],
    )
    if with_val:
        return train_labeled, train_unlabeled, val_dataset, test_dataset

    return train_labeled, train_unlabeled, test_dataset





def get_subset_fashion_mnist(
    num_classes=10,
    subset_size=20000,
    labeled_size=200,
    test_size=4000,
    transform_labeled=None,
    transform_unlabeled=None,
    seed=99,
    with_val=False,
    unsqueeze=True
):
    np.random.seed(seed)
    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
            transforms.ToTensor(),  # Convert the image to a tensor with pixel values in the range [0,1]
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize the pixel values using the mean and standard deviation of Fashion MNIST
        ]
    )

    # Load the datasets
    print(
        "getting Fashion MNIST | subset size, labeled size, test size =",
        [subset_size, labeled_size, test_size],
    )
    base_dataset = datasets.FashionMNIST(
        DATA_PATH, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        DATA_PATH, train=False, download=True, transform=transform
    )

    # --- Training data ---
    train_all_labels = np.array(base_dataset.targets)
    samples_per_class = subset_size // num_classes
    labeled_per_class = labeled_size // num_classes

    labeled_indices = []
    unlabeled_indices = []
    for i in range(num_classes):
        # get a subset of the original using indices
        class_indices = np.where(train_all_labels == i)[0]
        class_indices = np.random.choice(class_indices, samples_per_class, False)
        np.random.shuffle(class_indices)  # shuffle

        # split that subset to labeled and unlabeled
        labeled_indices.extend(class_indices[:labeled_per_class])
        unlabeled_indices.extend(class_indices[labeled_per_class:])

    train_labeled_indices = np.array(labeled_indices)
    train_unlabeled_indices = np.array(unlabeled_indices)

    # --- Testing data ---
    test_all_labels = np.array(test_dataset.targets)
    test_per_class = test_size // num_classes
    test_indices = []
    for i in range(num_classes):
        # get a subset of the original using indices
        class_indices = np.where(test_all_labels == i)[0]
        class_indices = np.random.choice(class_indices, test_per_class, False)
        np.random.shuffle(class_indices)  # shuffle
        test_indices.extend(class_indices)

    test_indices = np.array(test_indices)

    # --- Create dataset objects and return ---
    np.random.shuffle(train_labeled_indices)
    np.random.shuffle(train_unlabeled_indices)
    np.random.shuffle(test_indices)

    val_indices = test_indices[
        0 : int(test_size * 0.1)
    ]  # validation set is 10% of the test set

    test_indices = test_indices[int(test_size * 0.1) :]

    train_labeled = FashionMNIST_SSL(
        indices=train_labeled_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=True,
        unsqueeze=unsqueeze
    )
    train_unlabeled = FashionMNIST_SSL(
        indices=train_unlabeled_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=True,
        unsqueeze=unsqueeze
    )
    test_dataset = FashionMNIST_SSL(
        indices=test_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=False,
        unsqueeze=unsqueeze
    )

    val_dataset = FashionMNIST_SSL(
        indices=val_indices,
        root=DATA_PATH,
        transform=None,
        download=True,
        train=False,
        unsqueeze=unsqueeze
    )
    print(
        "labeled, unlabeled, test sizes:",
        [len(train_labeled), len(train_unlabeled), len(test_dataset)],
    )
    if with_val:
        return train_labeled, train_unlabeled, val_dataset, test_dataset

    return train_labeled, train_unlabeled, test_dataset




# Return an object of class torchvision.datasets
def get_subset_cifar10(
    num_classes=10,
    subset_size=20000,
    labeled_size=200,
    test_size=4000,
    transform_labeled=None,
    transform_unlabeled=None,
    seed=99,
    with_val=False,
    unsqueeze=True
):
    np.random.seed(seed)
    print(
        "getting cifar10 | subset size, labeled size, test size =",
        [subset_size, labeled_size, test_size],
    )
    base_dataset = datasets.CIFAR10(DATA_PATH, train=True, download=True)
    test_dataset = datasets.CIFAR10(DATA_PATH, train=False, download=True)

    # --- Training data ---
    train_all_labels = np.array(base_dataset.targets)
    samples_per_class = subset_size // num_classes
    labled_per_class = int(labeled_size // num_classes)

    labeled_indices = []
    unlabeled_indices = []
    for i in range(num_classes):
        # get a subset of the originial using indices
        class_indices = np.where(train_all_labels == i)[0]
        class_indices = np.random.choice(class_indices, samples_per_class, False)
        np.random.shuffle(class_indices)  # shuffle

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
        np.random.shuffle(class_indices)  # shuffle
        test_indices.extend(class_indices)

    test_indices = np.array(test_indices)

    # --- Create dataset objects and return ---
    np.random.shuffle(train_labeled_indices)
    np.random.shuffle(train_unlabeled_indices)
    np.random.shuffle(test_indices)

    if with_val:
        val_indices = test_indices[
            0 : int(test_size * 0.1)
        ]  # validation set is 10% of the test set
        test_indices = test_indices[int(test_size * 0.1) :]
        val_dataset = CIFAR10_SSL(
            val_indices, train=False, transform=None, download=True
        )

    train_labeled = CIFAR10_SSL(
        train_labeled_indices, train=True, transform=None, download=True
    )
    train_unlabeled = CIFAR10_SSL(
        train_unlabeled_indices, train=True, transform=None, download=True
    )
    test_dataset = CIFAR10_SSL(test_indices, train=False, transform=None, download=True)

    print(
        "labeled, unlabeled, test sizes:",
        [len(train_labeled), len(train_unlabeled), len(test_dataset)],
    )

    if with_val:
        return train_labeled, train_unlabeled, val_dataset, test_dataset

    return train_labeled, train_unlabeled, test_dataset


def get_poison_dataset_from_images(injected_folder):
    X_adv = []
    y_label = 99  # doesn't matter
    for fname in os.listdir(injected_folder):
        if "." in fname and "jpg" == fname.split(".")[-1]:
            im_path = os.path.join(injected_folder, fname)
            im = imageio.imread(im_path)
            X_adv.append(im)
    X_adv = np.array(X_adv)
    poison_dataset = CIFAR_POISON(data=X_adv, label=y_label, transform=None)
    return poison_dataset


# def transpose(x, source='NHWC', target='NCHW'):
#     print("Here",source == 'NCHW' and len(x.shape) == 4 and x.shape[1] == 3, x.shape)
#     # Check input shape
#     if source == 'NHWC' and len(x.shape) == 4 and x.shape[-1] == 3:
#         # Input shape is (batch_size, height, width, channels)
#         # Transpose to (batch_size, channels, height, width)
#         return x.transpose([0, 3, 1, 2])
#     elif source == 'NCHW' and len(x.shape) == 4 and x.shape[1] == 3:
#         # Input shape is (batch_size, channels, height, width)
#         # Transpose to (batch_size, height, width, channels)
#         return x.transpose([0, 2, 3, 1])
#     else:
#         # Input shape is not recognized, return original array
#         return x

cifar10_mean = (
    0.4914,
    0.4822,
    0.4465,
)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (
    0.2471,
    0.2435,
    0.2616,
)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

svhn_mean = (0.4376821, 0.4437697, 0.47280442)
svhn_std = (0.19803012, 0.20101562, 0.19703614)

food101_mean = (0.5, 0.5, 0.5)
food101_std = (0.5, 0.5, 0.5)


def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    mean = mean.reshape(3, 1, 1)  # reshape mean to (3, 1, 1)
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x

class FashionMNIST_SSL(datasets.FashionMNIST):
    def __init__(
        self,
        indices=None,
        root=None,
        transform=None,
        target_transform=None,
        download=True,
        train=True,
        asPIL = False,
        unsqueeze=True
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            train=train
        )
        self.asPIL = asPIL
        self.unsqueeze = unsqueeze
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.unsqueeze:
            img = img.unsqueeze(0).float()
        if self.asPIL:
            img = img.numpy()
            img = Image.fromarray(img)

        if self.transform is not None:
            # pdb.set_trace()
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST_SSL(datasets.MNIST):
    def __init__(
        self,
        indices=None,
        root=None,
        transform=None,
        target_transform=None,
        download=True,
        train=True,
        asPIL=False,
        unsqueeze=True
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            train=train
        )
        self.asPIL = asPIL
        self.unsqueeze = unsqueeze
        if indices is not None:
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.unsqueeze:
            img = img.unsqueeze(0).float()
        if self.asPIL:
            img = img.numpy()
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


import imageio
import os




def get_poison_dataset_from_images(injected_folder):
	X_adv = []
	y_label = 99 # doesn't matter	
	for fname in os.listdir(injected_folder):
		if '.' in fname and 'jpg' == fname.split('.')[-1]:
			im_path = os.path.join(injected_folder, fname)
			im = imageio.imread(im_path)
			X_adv.append(im)
	X_adv = np.array(X_adv)
	poison_dataset = CIFAR_POISON(data=X_adv, label=y_label, transform=None)
	return poison_dataset




# Custom dataset using a subset of the original specified by indices
class CIFAR10_SSL(datasets.CIFAR10):
    def __init__(
        self,
        indices,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        asPIL=False,
    ):
        super().__init__(
            "./data",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.asPIL = asPIL
        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]
        #   self.data = transpose(normalize(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print("IMG HAS SHAPEEE", img.shape)
        if self.asPIL:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR_POISON(Dataset):
    def __init__(self, data, label, transform=None, asPIL=False):
        self.data = data
        self.label = label
        self.transform = transform
        self.asPIL = asPIL
        self.targets = [label]*len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.label  # unlabeled, but keep for evaluation purposes

        if self.asPIL:
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
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
    a, b = next(combined_loader)
    print("a:", type(a))
    img = a.numpy()[0]
    label = b.numpy()[0]
    print(img)
    print(img.shape, type(img), type(img[0, 0, 0]))
    # print(b.numpy())
    # print("combined data has", i, "samples")

    # base_dataset = datasets.CIFAR10('./data', train=True, download=True)
    # base_dataset = base_dataset[[0,1,2]]
    # print(len(base_dataset))
    # print(base_dataset[0])

	




	

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

