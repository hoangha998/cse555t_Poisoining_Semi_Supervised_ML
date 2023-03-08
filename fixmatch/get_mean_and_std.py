# from data_getter import get_cifar10
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch

def main():

	dataset = datasets.Food101('./data', split='train', transform=transforms.ToTensor(), download=True)
	loader = DataLoader(
		dataset,
		batch_size=16,
		num_workers=1,
		drop_last=True)

	imgs = [item[0] for item in dataset] # item[0] and item[1] are image and its label
	imgs = torch.stack(imgs, dim=0).numpy()

	
	mean_r = imgs[:,0,:,:].mean()
	mean_g = imgs[:,1,:,:].mean()
	mean_b = imgs[:,2,:,:].mean()
	means = (mean_r, mean_g, mean_b)
	print("Mean:", means)

	
	std_r = imgs[:,0,:,:].std()
	std_g = imgs[:,1,:,:].std()
	std_b = imgs[:,2,:,:].std()
	stds = (std_r, std_g, std_b)
	print("Std:", stds)
	


if __name__ == '__main__':
    main()