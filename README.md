# Introduction
These are my personal notes while working through the [Dive into Deep Learning](https://d2l.ai) Book

## Development Environment
I created a seperate miniconda environment for this repo called "d2l", based on python version `3.9.18`.

The additionally added packages (mostly just linting stuff) are entered in the `requirements.txt`. This is the first time I'm using (mini)conda, next time I'm creating a repo that uses miniconda I'd create a local (mini)conda environment just how I'd do with a .venv file, but now that I works there is no need to reinstall everything just to have it localized. Might change my mind if linking errors appear in the future.

### Modifications of Libraries
Given that I'm the only person working on this repo and there is very likely not going to be any updates to the packages I think it's okay for me to directly modify the packages, mostly to remove things like deprecation and "Work In Progress" type warnings. This has the additional benefit that I don't have to suppress warnings on the module level but instead can deal with them one by one.

#### PyTorch
Inside of ```torch/nn/modules/lazy.py``` I've commented the lazy eval warning in line 180-181.

## Setup Code and Framework specific stuff
I will use [PyTorch](https://github.com/pytorch/pytorch) as the framework while
working through the book.

To get the (PyTorch) files you can run the `get_pytorch_files.sh` shell script.

# Licensing
This repository contains modified versions of the notebooks from the [Dive into Deep Learning](https://d2l.ai/) book, originally created by the D2L team.

The original notebooks are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-sa/4.0/), and my modifications are also licensed under the same CC BY-SA 4.0 license.

Attribution:
- The D2L Team
- https://github.com/d2l-ai/d2l-en

My modifications to the notebooks are indicated with comments or other appropriate markers.

Disclaimer: I have no experience with this kind of licensing attribution, so if something is mishandeled or missing, please let me know.