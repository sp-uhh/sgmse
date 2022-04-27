from math import ceil
import warnings

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import wandb

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.validation_tools import visualize_process


class ScoreModel(pl.LightningModule):
    def __init__(self,
        backbone: str, sde: str,
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, reduce_mean: bool = False,
        transform: str = 'none', input_y: bool = True, nolog: bool = False,
        weighting_exponent: float = 0.0, g_weighting_exponent: float = 0.0, output_std_exponent: float = -1.,
        loss_type: str = 'mse', data_module_cls = None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        self.input_y = input_y
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        if self.input_y:
            ch = kwargs.get('input_channels', None)
            if ch is not None and ch != 2:
                warnings.warn("Overriding input_channels from {ch} to 2 since 'input_y' is set")
            kwargs.update(input_channels=2)
        print(self.input_y)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.output_std_exponent = output_std_exponent
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.reduce_mean = reduce_mean
        self.weighting_exponent = weighting_exponent
        self.g_weighting_exponent = g_weighting_exponent
        self.loss_type = loss_type

        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        # Construct the reduce-operation function for the loss calculation
        self._reduce_op = torch.mean if self.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema-decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t-eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--reduce-mean", action="store_true", help="Average loss across all data dimensions (as opposed to summing)")
        parser.add_argument("--input-y", dest='input_y', action="store_true", help="Provide y to the score model")
        parser.add_argument("--no-input-y", dest='input_y', action="store_false", help="Don't provide y to the score model")
        parser.set_defaults(input_y=True)
        parser.add_argument("--output-std-exponent", type=float, default=0.0, help="Weight model output by (std**exponent). Default is 0.")
        parser.add_argument("--weighting-exponent", type=float, default=0.0, help="The exponent for the loss weighting (lambda=std**exponent). Can be combined with --g-weighting-exponent.")
        parser.add_argument("--g-weighting-exponent", type=float, default=0.0, help="The exponent for g in the loss weighting (lambda=g**exponent). Can be combined with --weighting-exponent.")
        parser.add_argument("--loss-type", type=str, default="mse", choices=("mse", "mae", "gaussian_entropy"), help="The type of loss function to use.")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, sigmas, gs):
        weighting = sigmas**self.weighting_exponent * gs**self.g_weighting_exponent

        if self.loss_type == 'gaussian_entropy':
            cov = self._weighted_mean(err.abs()**2, w=weighting)**2
            pcov = torch.abs(self._weighted_mean(err**2, w=weighting))**2
            loss = cov - pcov
            return loss

        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        losses = losses * weighting
        # sum loss for each pixel
        losses = self._reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def _step(self, batch, batch_idx):
        # `x` is a clean complex spectrogram, `y` is a (non-Gaussian-)noisy complex spectrogram.
        x, y = batch
        t = torch.rand(y.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        # `mean` is the current OU SDE mean, interpolated between x and y.
        mean, std = self.sde.marginal_prob(x, t, y)
        # `perturbed_data` is a direct sample from the (solved) forward process of the SDE.
        z = torch.randn_like(y)
        sigmas = std[:, None, None, None]
        gs = self.sde.sde(x, t[:, None, None, None], y)[1]
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y)
        err = score * sigmas + z
        loss = self._loss(err, sigmas, gs)
        return loss

    # TODO check if this function works as intended for the SE input and potentially fix
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        if batch_idx == 0 and self.local_rank == 0:  # make visualization logs only from rank zero process
            x, y = batch
            # fig_dict = visualize_process(x, y, model=self, N=50, M=min(3, x.shape[0]))
            # for name, fig in fig_dict.items():
            #     if self.nolog:
            #         pass
            #         # fig.savefig(f'figures/{name}_{self.current_epoch}.png')
            #     else:
            #         pass
            #         #self.logger.log_image(key=name, images=[fig])
            #     # plt.close(fig)
        return loss

    def forward(self, x, t, y):
        # not sure if the minus and scaling is important here - taken from Song for VPSDE case
        score = -self._raw_dnn_output(x, t, y)
        std = self.sde._std(t)
        score = score * std[:, None, None, None]**self.output_std_exponent
        return score

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def _raw_dnn_output(self, x, t, y):
        """only for debugging"""
        if self.input_y == True:
            dnn_input = torch.cat([x, y], dim=1)
        else:
            dnn_input = x
        return self.dnn(dnn_input, t)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)