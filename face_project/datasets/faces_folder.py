import cv2
from pathlib import Path
from torch.utils.data import Dataset


class FacesFolder(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.image_files = list(self.folder.glob('*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
