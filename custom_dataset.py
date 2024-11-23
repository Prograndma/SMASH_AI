from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class CustomDataset(Dataset):
    def __init__(self, dataset, culled):
        self.dataset = dataset
        self.culled = culled
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        if self.culled:
            targets = torch.tensor([
                # sample["Start"],
                sample["A"],
                sample["B"],
                sample["X"],
                # sample["Y"],
                sample["Z"],
                # sample["DPadUp"],
                # sample["DPadDown"],
                # sample["DPadLeft"],
                # sample["DPadRight"],
                # sample["L"],
                # sample["R"],
                sample["LPressure"] / 255,
                sample["RPressure"] / 255,
                sample["XAxis"] / 255,
                sample["YAxis"] / 255,
                sample["CXAxis"] / 255,
                sample["CYAxis"] / 255],
                dtype=torch.float32)
        else:
            targets = torch.tensor([
                sample["Start"],
                sample["A"],
                sample["B"],
                sample["X"],
                sample["Y"],
                sample["Z"],
                sample["DPadUp"],
                sample["DPadDown"],
                sample["DPadLeft"],
                sample["DPadRight"],
                sample["L"],
                sample["R"],
                sample["LPressure"] / 255,
                sample["RPressure"] / 255,
                sample["XAxis"] / 255,
                sample["YAxis"] / 255,
                sample["CXAxis"] / 255,
                sample["CYAxis"] / 255],
                dtype=torch.float32)

        return {'image': image, 'targets': targets}
