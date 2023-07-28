import pickle
import tfrecord_dataset as tfr
from tfrecord_dataset.torch import TFRecordDataset
from tfrecord_dataset.tools.tfrecord2idx import create_index
from methods.abstract import DatasetTester

class TFRecordTester(DatasetTester):
    def _initialize_files(self, source_torch_dataset, dstdir):
        dstdir, self.dstfile = self.init_dir(dstdir, filename='data.tfrecord')
        writer = tfr.TFRecordWriter(str(self.dstfile))
        for ind, sample in enumerate(source_torch_dataset):
            sample_bytes = pickle.dumps(sample)
            writer.write(sample_bytes)
        writer.close()
        # create index
        self.index_path = self.dstfile.parent / (self.dstfile.stem + '.index')
        create_index(str(self.dstfile), str(self.index_path))

    def get_torch_dataset(self):
        return TFRecordDataset(str(self.dstfile), str(self.index_path))

    def get_name(self):
        return 'tfrecord'
