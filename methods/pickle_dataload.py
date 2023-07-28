import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from methods.abstract import DatasetTester

class PickleDataset(Dataset):
    def __init__(self, dstdir, nchars, pickle_load_fn):
        super().__init__()
        self.dstdir = dstdir
        self.nchars = nchars
        self.pickle_load = pickle_load_fn

    def __getitem__(self, index):
        with open(self.dstdir / f'{str(index).zfill(self.nchars)}.pickle', 'rb') as f:
            sample = self.pickle_load(f)
        return sample

    def __len__(self):
        return len(list(self.dstdir.glob('*.pickle')))

class PickleTester(DatasetTester):
    def __init__(self, num_workers=0):
        self.num_workers = num_workers

    def _initialize_files(self, source_torch_dataset, dstdir):
        self.dstdir, _ = self.init_dir(dstdir, filename='')
        dl = DataLoader(source_torch_dataset, batch_size=1, num_workers=self.num_workers)
        print(f'creating pickle dataset of size {len(source_torch_dataset)}...')
        self.nchars = int(np.ceil(np.log10(len(dl))))
        for ind, sample in enumerate(dl):
            with open(self.dstdir / f'{str(ind).zfill(self.nchars)}.pickle', 'wb') as f:
                self.pickle_dump(sample, f)
    
    def pickle_dump(self, sample, f):
        pickle.dump(sample, f)

    def pickle_load(self, f):
        return pickle.load(f)

    def get_torch_dataset(self):
        return PickleDataset(self.dstdir, self.nchars, self.pickle_load)

    def get_name(self):
        return 'pickle'

        