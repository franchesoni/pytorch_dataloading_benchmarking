from methods.abstract import DatasetTester

class DefaultTester(DatasetTester):
    def _initialize_files(self, source_torch_dataset, dstdir):
        self.source_torch_dataset = source_torch_dataset

    def get_torch_dataset(self):
        return self.source_torch_dataset

    def get_name(self):
        return 'default'