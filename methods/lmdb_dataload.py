import lmdb
import pickle
from torch.utils.data import DataLoader, Dataset

from methods.abstract import DatasetTester

class LMDBDataset(Dataset):
    def __init__(self, dstfile):
        super().__init__()
        self.dstfile = dstfile

    def open_lmdb(self):
         self.env = lmdb.open(str(self.dstfile), subdir=False,
                              readonly=True, lock=False,
                              readahead=False, meminit=False, max_readers=1)
         self.txn = self.env.begin(write=False, buffers=True)
         self.length = pickle.loads(self.txn.get(b'__len__'))
         self.keys = pickle.loads(self.txn.get(b'__keys__'))

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        byteflow = self.txn.get(self.keys[index])
        sample = pickle.loads(byteflow)
        return sample

    def __len__(self):
        if not hasattr(self, 'length'):
            env = lmdb.open(str(self.dstfile), subdir=False,
                            readonly=True, lock=False,
                            readahead=False, meminit=False, max_readers=1)
            with env.begin(write=False) as txn:
                self.length = pickle.loads(txn.get(b'__len__'))
                self.keys = pickle.loads(txn.get(b'__keys__'))
        return self.length


class LMDBTester(DatasetTester):
    def __init__(self, write_frequency=1000, num_workers=0, map_size_in_bytes=1024**3):
        self.write_frequency = write_frequency
        self.num_workers = num_workers
        self.map_size = map_size_in_bytes

    def _initialize_files(self, source_torch_dataset, dstdir):
        filename = 'lmdbtester.lmdb'
        dstdir, dstfile = self.init_dir(dstdir, filename)
        self.dstfile = dstfile  # save for later
        data_loader = DataLoader(source_torch_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

        print("Generating LMDB to %s" % dstfile)
        db = lmdb.open(str(dstfile), subdir=dstfile.is_dir(),
                    meminit=False, map_async=True, writemap=True,
                    map_size=self.map_size)

        print(f'creating LMDB dataset of size {len(source_torch_dataset)}...')
        txn = db.begin(write=True)
        for idx, sample in enumerate(data_loader):
            sample = sample[0]
            txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps(sample))
            if idx % self.write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(len(data_loader))]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
            txn.put(b'__len__', pickle.dumps(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()

    def get_torch_dataset(self) -> Dataset:
        return LMDBDataset(self.dstfile)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.dstfile) + ')'

    def get_name(self):
        return 'LMDB'
