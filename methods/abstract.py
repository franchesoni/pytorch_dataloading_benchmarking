from abc import ABC, abstractmethod
import torch
import time
import os
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import resource

# define a decorator that limits memory usage
def limit_memory (maxsize):
    def decorator (func):
        def wrapper (*args, **kwargs):
            # get the current memory limit
            soft, hard = resource.getrlimit (resource.RLIMIT_AS)
            # set the new memory limit
            resource.setrlimit (resource.RLIMIT_AS, (maxsize, hard))
            try:
                # execute the function
                result = func (*args, **kwargs)
            except RuntimeError as e:
                if 'memory' in str(e):
                    print('Memory error!')
                    result = None
                else:
                    raise e
            # restore the original memory limit
            resource.setrlimit (resource.RLIMIT_AS, (soft, hard))
            # return the result
            return result
        return wrapper
    return decorator

# use the decorator on your function
@limit_memory (1024 * 1024 * 1024) # 1 GB
def my_function ():
    # do something that may use a lot of memory
    pass


class DatasetTester(ABC):
    @abstractmethod
    def _initialize_files(self, source_torch_dataset, dstdir):
        pass

    @abstractmethod
    def get_torch_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @limit_memory(1024 * 1024 * 1024 * 8)  # 4 GB
    def initialize_files(self, source_torch_dataset, dstdir):
        print('-'*20, 'Initializing', self.get_name(), '-'*20)
        self.initialized = True
        st = time.time()
        self._initialize_files(source_torch_dataset, dstdir)
        print(f'{self.get_name()} initializing time:', time.time() - st)

    def test(self, num_workers=0, batch_size=1, prefetch_factor=None, shuffle=False, pin_memory=False, drop_last=False):
        if not self.initialized:
            raise RuntimeError('Must call initialize_files before testing.')
        print('-'*20, 'Testing', self.get_name(), '-'*20)
        dataset = self.get_torch_dataset()
        dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last)
        st = time.time()
        for ind, sample in enumerate(dl):
            pass
        print(f'{self.get_name()} loading time:', time.time() - st)

    def init_dir(self, dstdir, filename):
        dstdir = Path(dstdir)
        dstfile = dstdir / filename
        if dstfile.exists():
            print('Warning: Deleting existing file at %s' % dstfile)
            if os.path.isdir(dstfile):
                shutil.rmtree(dstfile)
            else:
                os.remove(dstfile)
                if os.path.exists(str(dstfile) + '-lock'):
                    os.remove(str(dstfile) + '-lock')
        dstdir.mkdir(exist_ok=True, parents=True)
        return dstdir, dstfile
