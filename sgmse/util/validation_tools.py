import numpy as np
import matplotlib.pyplot as plt, matplotlib.patheffects as PathEffects
import torch
from pesq import pesq


def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def get_img(cspec, db=False):
    out = torch.abs(cspec[0,0,:,:]).detach().cpu().numpy()
    if db:
        out = 20 * np.log10(out + 1e-12)
    return out


def _power(x):
    return torch.sum(x.view(-1).abs()**2)


def get_line_plot(debug_results, dpi=200):
    ts = np.array(debug_results["t"])

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=dpi)
    ax = axs[0]
    mse_x = debug_results["mse_x"]
    mse_y = debug_results["mse_y"]
    ax.set_xlabel('t')
    ax.set_ylabel('MSE')
    ax.plot(ts, mse_x, '.-', label="MSE(x0, xt)")
    ax.plot(ts, mse_y, '.-', label="MSE(y, xt)")
    ax.set_title('MSEs')
    ax.legend()

    ax = axs[1]
    p_dsde = [_power(d[0,0]) for d in debug_results["sde_drift"]]
    p_dscore = [_power(d[0,0]) for d in debug_results["score_drift"]]
    p_dtotal = [
        _power(dsde[0,0]) / _power(dsde[0,0]+dsc[0,0])
        for dsde, dsc in zip(debug_results["sde_drift"], debug_results["score_drift"])
    ]
    ax.set_xlabel('t')
    ax.set_ylabel('Drift power')
    tax = ax.twinx()
    lines = []
    lines.append(ax.plot(ts, p_dsde, '.-')[0])
    lines.append(ax.plot(ts, p_dscore, '.-')[0])
    lines.append(tax.plot(ts, p_dtotal, '.-', color='k')[0])
    tax.legend(handles=lines, labels=("SDE", "Score", "a/a+b"))
    ax.set_title('Drift power')
    fig.tight_layout()
    return fig


def get_spec_plot(x, y, debug_results, db=False, dpi=200, backward_transform=None):
    if db:
        vmin, vmax = -30, 25
    else:
        vmin, vmax = 0, 1

    if backward_transform is None:
        backward_transform = lambda x: x

    ts = [1, .9, .8, .7, .5, .3, .2, .1, .0]
    sampled_ts = np.array(debug_results["t"])
    nearest_idxs = [find_nearest_idx(sampled_ts, t) for t in ts]
    titles = ["y", "xT", *ts, "x"]
    shown_xs = [y, debug_results["xT"]] + [debug_results["xt"][nearest_idx] for nearest_idx in nearest_idxs] + [x]

    fig, axs = plt.subplots(3, 4, figsize=(7, 5), sharex=True, sharey=True, dpi=dpi)
    for ax, title, shown in zip(axs.flat, titles, shown_xs):
        shown = backward_transform(shown)
        im = ax.imshow(get_img(shown, db=db), vmin=vmin, vmax=vmax, origin="lower", cmap="magma", aspect="equal")
        text = ax.text(x=0.05, y=0.85, s=title, transform=ax.transAxes)
        pe = [PathEffects.withStroke(linewidth=3, foreground="w")]
        text.set_path_effects(pe)
    fig.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
    return fig


def get_pesq_plot(x, y, debug_results, model, M):
    ts = [1, .9, .8, .7, .5, .3, .2, .1, .0]
    sampled_ts = np.array(debug_results["t"])
    nearest_idxs = [find_nearest_idx(sampled_ts, t) for t in ts]
    compared_xs = [y, debug_results["xT"]] + [debug_results["xt"][nearest_idx] for nearest_idx in nearest_idxs]
    compared_sigss = [model._istft(model._backward_transform(x_).squeeze(1)).cpu().numpy() for x_ in compared_xs]
    true_sigs = model._istft(model._backward_transform(x).squeeze(1)).cpu().numpy()
    noisy_sigs = model._istft(model._backward_transform(y).squeeze(1)).cpu().numpy()

    mos_lqos = [[] for _ in range(M)]
    mos_lqos_mix08 = [[] for _ in range(M)]
    for i in range(M):
        for compared_sig in compared_sigss:
            if true_sigs[i].shape != compared_sig[i].shape or true_sigs[i].shape == ():
                raise ValueError(
                    f"mismatching or zero-dimensional shapes of compared_sig[{i}] {compared_sig[i].shape} "
                    f"and true_sigs[{i}] {true_sigs[i].shape}"
                )
            mos_lqo = pesq(16000, true_sigs[i], compared_sig[i], 'wb')
            mos_lqo_mix08 = pesq(16000, true_sigs[i], 0.8*compared_sig[i]+0.2*noisy_sigs[i], 'wb')
            mos_lqos[i].append(mos_lqo)
            mos_lqos_mix08[i].append(mos_lqo_mix08)

    fig = plt.figure()
    ax = fig.gca()
    # use t=1.2 and t=1.1 for y and xT, respectively
    for i in range(M):
        ax.plot([1.2, 1.1]+ts, mos_lqos[i], '.-', color=f'C{i}')
        ax.plot([1.2, 1.1]+ts, mos_lqos_mix08[i], '--', color=f'C{i}')
    ax.set_xlabel('t')
    ax.set_ylabel('PESQ MOS-LQO')
    fig.tight_layout()
    return fig


def visualize_process(x, y, model, N=100, M=3, pesq_only=False):
    x, y = x[:M].detach(), y[:M].detach()
    single_x, single_y = x[[0]], y[[0]]

    sampler = model.get_pc_sampler(
        "euler_maruyama", "none", y, N=N,
        probability_flow=False, denoise=False, debug=True, debug_x=x
    )
    prob_flow_sampler = model.get_pc_sampler(
        "euler_maruyama", "none", y, N=N,
        probability_flow=True, denoise=False, debug=True, debug_x=x
    )
    _, _, debug_results = sampler()
    #_, _, debug_results_pf = prob_flow_sampler()

    figs = {
        "PESQs": get_pesq_plot(x, y, debug_results, model, M=M),
    }
    if not pesq_only:
        figs = {
            **figs,
            "Lines": get_line_plot(debug_results, dpi=100),
            "Specs (transformed)": get_spec_plot(
                single_x, single_y, debug_results, dpi=200
            ),
            "Specs (dB, natural)": get_spec_plot(
                single_x, single_y, debug_results, dpi=200,
                db=True, backward_transform=model._backward_transform
            ),
            # "Specs (PF)": get_spec_plot(x, y, debug_results_pf, dpi=200),
            # "Specs (PF)": get_spec_plot(x, y, debug_results_pf, dpi=200, db=True),
        }
    return figs
