import sys
import torch
import cProfile
from pathlib import Path
from torch.utils.data import Dataset

from methods import LMDBTester, PickleTester, DefaultTester, DatasetTester, InmemoryTester

class ExampleDataset(Dataset):
    def __init__(self, N):
        self.strlen = len(str(N))
        self.data_indices = list(range(N))
        self.seed_offset = int(torch.randint(0, 1000000, (1,)))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError  # we need this to avoid infinite loop
        torch.manual_seed(idx + int(torch.randint(0, 1000000, (1,))))
        strind = str(idx).zfill(self.strlen)
        tensor4d, tensor2d = torch.rand(8, 9, 256, 256), torch.rand(256, 256)
        return strind, tensor4d, tensor2d




def main(base_dir, num_workers=0, prefetch_factor=None, batch_size=1, n_samples=10, spawn=False, tee=False):
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True, parents=False)
    if tee:
        stdoutOrigin = sys.stdout
        sys.stdout = open(base_dir / 'results.txt', 'w')

    print(f"base_dir: {base_dir}, num_workers: {num_workers}, prefetch_factor: {prefetch_factor}, batch_size: {batch_size}, n_samples: {n_samples}, spawn: {spawn}")
    print('='*80)
    if spawn:
        torch.multiprocessing.set_start_method('spawn')
    
    testers: list[DatasetTester] = [
        LMDBTester(write_frequency=100, num_workers=num_workers),
        DefaultTester(),
        InmemoryTester(),
        PickleTester(num_workers=num_workers),
    ]
    torch_dataset = ExampleDataset(n_samples)


    for tester in testers:
        print('testing method:', tester.get_name())
        profiler = cProfile.Profile()
        profiler.enable()
        tester.initialize_files(torch_dataset, base_dir / tester.get_name())
        print('first...')
        tester.test(num_workers=num_workers, batch_size=batch_size, prefetch_factor=prefetch_factor)
        print('second...')
        tester.test(num_workers=num_workers, batch_size=batch_size, prefetch_factor=prefetch_factor)
        print('='*80)
        profiler.disable()
        profiler.dump_stats(base_dir / (tester.get_name() + '.prof'))

    if tee:
        sys.stdout.close()
        sys.stdout = stdoutOrigin

    

if __name__=='__main__':
    import fire
    fire.Fire(main)