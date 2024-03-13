# Introduction
These are my personal notes while working through the [Dive into Deep Learning](https://d2l.ai) Book

## Development Environment
I created a seperate miniconda environment for this repo called "d2l", based on python version `3.9.18`.

The additionally added packages (mostly just linting stuff) are entered in the `requirements.txt`. This is the first time I'm using (mini)conda, next time I'm creating a repo that uses miniconda I'd create a local (mini)conda environment just how I'd do with a .venv file, but now that I works there is no need to reinstall everything just to have it localized. Might change my mind if linking errors appear in the future.

## Setup Code and Framework specific stuff
I will use [PyTorch](https://github.com/pytorch/pytorch) as the framework while
working through the book.

To get the (PyTorch) files you can run the `get_pytorch_files.sh` shell script.