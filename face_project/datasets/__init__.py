from torch.utils.data import Dataset

from .celeba import CelebA
from .faces_folder import FacesFolder


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])
