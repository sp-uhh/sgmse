# Speech Enhancement and Dereverberation with Diffusion-based Generative Models

<img src="https://raw.githubusercontent.com/sp-uhh/sgmse/main/diffusion_process.png" width="500" alt="Diffusion process on a spectrogram: In the forward process noise is gradually added to the clean speech spectrogram x0, while the reverse process learns to generate clean speech in an iterative fashion starting from the corrupted signal xT.">

This repository contains the official PyTorch implementations for the 2022 papers:

- Simon Welker, Julius Richter, Timo Gerkmann. [*"Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain"*](https://www.isca-speech.org/archive/interspeech_2022/welker22_interspeech.html), ISCA Interspeech, Incheon, Korea, Sep. 2022. [[bibtex]](#citations--references)
- Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. [*"Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"*](https://ieeexplore.ieee.org/abstract/document/10149431), IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. [[bibtex]](#citations--references)

Audio examples and further supplementary materials are available [on our project page](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/sgmse).

## Follow-up work

Please also check out our follow-up work with code available: 

- Jean-Marie Lemercier, Julius Richter, Simon Welker, Timo Gerkmann. [*"StoRM: A Diffusion-based Stochastic Regeneration Model for Speech Enhancement and Dereverberation"*](https://arxiv.org/abs/2212.11851), submitted to IEEE/ACM Trans. on Audio, Speech, Language Proc. (TASLP). [[github]](https://github.com/sp-uhh/storm)
- Bunlong Lay, Simon Welker, Julius Richter, Timo Gerkmann. [*"Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement"*](https://arxiv.org/abs/2302.14748), ISCA Interspeech, Dublin, Ireland, Aug. 2023. [[github]](https://github.com/sp-uhh/sgmse-bbed)

## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.


## Pretrained checkpoints

- For the Speech Enhancement task, we provide pretrained checkpoints for the models trained on VoiceBank-DEMAND and WSJ0-CHiME3, as in the paper. They can be downloaded [here](https://drive.google.com/drive/folders/1CSnkhUSoiv3RG0xg7WEcVapyLuwDaLbe?usp=sharing).
- For the Dereverberation task, we provide a checkpoint trained on our WSJ0-REVERB dataset. It can be downloaded [here](https://drive.google.com/drive/folders/1082_PSEgrqoVVrNsAkSIcpLF1AAtzGwV?usp=sharing).
    - Note that this checkpoint works better with sampler settings `--N 50 --snr 0.33`.

Usage:
- For resuming training, you can use the `--resume_from_checkpoint` option of `train.py`.
- For evaluating these checkpoints, use the `--ckpt` option of `enhancement.py` (see section **Evaluation** below).


## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [2]) can be run with

```bash
python train.py --base_dir <your_base_dir>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.

To see all available training options, run `python train.py --help`. Note that the available options for the SDE and the backbone network change depending on which SDE and backbone you use. These can be set through the `--sde` and `--backbone` options.

**Note:**
- Our journal preprint [2] uses `--backbone ncsnpp`.
- Our Interspeech paper [1] uses `--backbone dcunet`. You need to pass `--n_fft 512` to make it work.
    - Also note that the default parameters for the spectrogram transformation in this repository are slightly different from the ones listed in the first (Interspeech) paper (`--spec_factor 0.15` rather than `--spec_factor 0.333`), but we've found the value in this repository to generally perform better for both models [1] and [2].


## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
```

to generate the enhanced .wav files, and subsequently run

```bash
python calc_metrics.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir>
```

to calculate and output the instrumental metrics.

Both scripts should receive the same `--test_dir` and `--enhanced_dir` parameters. The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.


## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:
```bib
@inproceedings{welker22speech,
  author={Simon Welker and Julius Richter and Timo Gerkmann},
  title={Speech Enhancement with Score-Based Generative Models in the Complex {STFT} Domain},
  year={2022},
  booktitle={Proc. Interspeech 2022},
  pages={2928--2932},
  doi={10.21437/Interspeech.2022-10653}
}
```
```bib
@article{richter2023speech,
  title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
  author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={2351-2364},
  year={2023},
  doi={10.1109/TASLP.2023.3285241}
}
```

>[1] Simon Welker, Julius Richter, Timo Gerkmann. "Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain", ISCA Interspeech, Incheon, Korea, Sep. 2022.
>
>[2] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023.