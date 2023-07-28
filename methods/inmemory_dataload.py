from torch.utils.data import Dataset

from methods.abstract import DatasetTester

class InmemoryTester(DatasetTester):
    def _initialize_files(self, source_torch_dataset, dstdir):
        self.samples = []
        for ind, sample in enumerate(source_torch_dataset):
            self.samples.append(sample)

    def get_torch_dataset(self):
        return InmemoryDataset(self.samples)

    def get_name(self) -> str:
        return 'inmemory'
        
class InmemoryDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)