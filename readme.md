
# Dataloading for PyTorch

How many times dataloading was your bottleneck? Many?

Dataloading has to be parallel and just a little faster than your training/validation steps. After you reach very high GPU use there is little point on further optimizing. However, how do we make dataloading fast enough?

Looking around there are many solutions:
- FFCV, whose installation is hell
- default pytorch's `Dataloader`, which we'll use but isn't enough by itself
- using cPickle, that is recommended but makes no sense in Python3 (it's part of the standard library and `pickle` uses it)
- tensorpack's dataflow which looks simultaneously promising and very old
- formats such as LMDB, TFRecords, HDF5, that are `single-file` and allow us to not skip from file to file

## What this is
There is an example dataset that generates random tensors with the format of the samples in contrail-detection competition (4D tensor, 2D tensor, string)

## How to use
- implement your method under `methods/` implementing `initialize_files(self, source_torch_dataset, dstdir)`, `get_torch_dataset(self)`, `get_name(self)` as in the examples.
- put it in `methods/__init__.py`
- load it in `main.py` and add it to the list of methods to compare
- you can now use `python main.py --help` to see what the parameters are
- obviously you can launch many experiments if you write them as lines in `experiments.sh` (see the file for command examples)