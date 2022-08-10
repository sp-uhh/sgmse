# Speech Enhancement and Dereverberation with Diffusion-based Generative Models

This repository contains the official PyTorch implementation for the 2022 paper *Speech Enhancement and Dereverberation with Diffusion-based Generative Models* by Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay and Timo Gerkmann.


## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- Set up a [wandb.ai](https://wandb.ai/) account, or disable this logging by always calling `train.py` with the `--nolog` option.
- If using W&B logging, log in via `wandb login` before running our code.


## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper) can be run with
```bash
python train.py --base_dir <your_base_dir>
```
where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently support training only with `.wav` files.

To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.


## Evaluation

TODO


## Citations

We kindly ask you to cite our papers in your publication when using any of our research or code:

>Simon Welker, Julius Richter and Timo Gerkmann. *Speech enhancement with score-based generative models in the complex STFT domain*, ISCA Interspeech, 2022.
>Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay and Timo Gerkmann. *Speech Enhancement and Dereverberation with Diffusion-based Generative Models*, TBD, 2022.