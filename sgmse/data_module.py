
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(
        self, data_dir, subset, dummy, shuffle_spec, num_frames,
        normalize_audio=True, spec_transform=None, stft_kwargs=None, **ignored_kwargs
    ):
        self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
        self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize_audio = normalize_audio
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        s, _ = load(self.clean_files[i])
        x, _ = load(self.noisy_files[i])
        normfac = s.abs().max()

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = s.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            s = s[..., start:start+target_len]
            x = x[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            s = F.pad(s, (pad//2, pad//2+(pad%2)), mode='constant')
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')

        if self.normalize_audio:
            # normalize both based on clean speech, to ensure same clean signal power in s and x.
            # this means that |x|>1 is possible, which is okay for the float32 domain we work in.
            s = s / normfac
            x = x / normfac

        S = torch.stft(s, **self.stft_kwargs)
        X = torch.stft(x, **self.stft_kwargs)

        S, X = self.spec_transform(S), self.spec_transform(X)
        return S, X

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/40)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    def __init__(
        self, base_dir, batch_size=32,
        n_fft=512, hop_length=128, num_frames=256, window='sqrthann',
        num_workers=4, dummy=False, spec_factor=1, spec_abs_exponent=1,
        gpu=True, **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
            **self.stft_kwargs, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(self.base_dir, 'train', self.dummy, True, **specs_kwargs)
            self.valid_set = Specs(self.base_dir, 'valid', self.dummy, False, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(self.base_dir, 'test', self.dummy, False, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.spec_abs_exponent != 1:
            # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
            # and introduced numerical error
            e = self.spec_abs_exponent
            spec = spec.abs()**e * torch.exp(1j * spec.angle())
        return spec * self.spec_factor

    def spec_back(self, spec):
        spec = spec / self.spec_factor
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base-dir", type=str, required=True,
            help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, "
                "each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--batch-size", type=int, default=32,
            help="The batch size. 32 by default.")
        parser.add_argument("--n-fft", type=int, default=512,
            help="Number of FFT bins. 512 by default.")
        parser.add_argument("--hop-length", type=int, default=128,
            help="Window hop length. 128 by default.")
        parser.add_argument("--num-frames", type=int, default=256,
            help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann",
            help="The window function to use for the STFT. 'sqrthann' by default.")
        parser.add_argument("--num-workers", type=int, default=4,
            help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true",
            help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec-factor", type=float, default=0.33,
            help="Factor to multiply complex STFT coefficients by. 1 by default (no effect).")
        parser.add_argument("--spec-abs-exponent", type=float, default=0.5,
            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). "
                "1 by default; set to values < 1 to bring out quieter features.")
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
