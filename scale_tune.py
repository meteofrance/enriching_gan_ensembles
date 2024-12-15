import os
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import timedelta
from glob import glob
from itertools import product
from random import shuffle, uniform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import perturbation.pca_stylegan as pca
import perturbation.smpca as smpca
import perturbation.utils as utils

from gan.model.stylegan2 import Generator

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def add_noise(betas, sig, device):
    noise = torch.empty(betas.shape).normal_()
    return betas + sig * noise.to(device)


def sigma(t, epoch):
    return (0.2 / (1 + epoch)) * (1 - t**2)


def learning_rate(t, lr0):
    if t < 0.25:
        return t * 0.01 + lr0
    elif t > 0.5:
        return 0.01 * (1.0 - t)
    return 0.01


def convert_uvt2fft(batch_gen, batch_y):
    new_batch_gen = torch.cat(
        (
            torch.sqrt(batch_gen[:, 0:1, :, :] ** 2 + batch_gen[:, 1:2, :, :] ** 2),
            batch_gen[:, 2:, :, :],
        ),
        dim=1,
    )
    new_batch_y = torch.cat(
        (
            torch.sqrt(batch_y[:, 0:1, :, :] ** 2 + batch_y[:, 1:2, :, :] ** 2),
            batch_y[:, 2:, :, :],
        ),
        dim=1,
    )

    return new_batch_gen, new_batch_y


parser = ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--n_samples", type=int, default=16)
parser.add_argument("--inflate_random", action="store_true")
parser.add_argument("--lr0", type=float, default=0.001)
parser.add_argument("--beta_rule", type=str, default="sigmoid")
parser.add_argument("--style_cut", type=int, default=10)
parser.add_argument("--inflate", type=float, default=1.0)
parser.add_argument("--start", type=str, default="ones")
parser.add_argument("--lambda_bias", type=float, default=1.0)
parser.add_argument("--lambda_spread", type=float, default=1.0)
parser.add_argument("--convert_ff_t", action="store_true")
parser.add_argument("--invstep", type=int, default=1000)
parser.add_argument(
    "--dates_file",
    type=str,
    default="./exemple.csv",
)
parser.add_argument("--date_start", type=str, default="")
parser.add_argument("--date_stop", type=str, default="1")
parser.add_argument(
        "--leadtimes",
        type=utils.str2intlist,
        default=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
    )

parser.add_argument(
    "--fake_data_dir",
    type=str,
    default="./",
)
parser.add_argument(
    "--real_data_dir",
    type=str,
    default="./",
)
parser.add_argument(
    "--pack_dir",
    type=str,
    default="./",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="./",
)
parser.add_argument(
    "--eigen_dir",
    type=str,
    default="./",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./",
)

args = parser.parse_args()


# making folders for experiment
output_dir = f"{args.output_dir}alphas_betas_pca_{args.style_cut}_{args.inflate_random}_{args.inflate}_bias_{args.start}_{args.lambda_bias}_spread_{args.lambda_spread}_ff_{args.convert_ff_t}_{args.invstep}/"
os.makedirs(output_dir, exist_ok=True)
# creating one new folder each time the experiment is relaunched
instances = len(glob(output_dir + "Instance_*/"))
print("instances already existing", instances)
os.makedirs(output_dir + f"Instance_{instances+1}/", exist_ok=True)
output_dir = output_dir + f"Instance_{instances+1}/"

# performing experiment on selected data
 ################## selecting dates
print("reading dates")
df = pd.read_csv(args.real_data_dir + args.dates_file)
df_date = df.copy()
df_date["Date"] = pd.to_datetime(df_date["Date"])
df_extract = df_date[
        (df_date["Date"] >= args.date_start) &  (args.date_stop >= df_date["Date"] - timedelta(days=1))
    ]
liste_dates = df_extract["Date"].unique()
# choosing on which lead times to optimize
leadtimes = args.leadtimes

ensemble_dataset = list(product(liste_dates, leadtimes))


# loading decorrelation matrix and average latent
Whitening = torch.load(args.eigen_dir + "Whitening.pt").to(device)
w0 = torch.load(args.eigen_dir + "latent_mean.pt").to(device)

print("loading G")

G = Generator(256, 512, n_mlp=8, nb_var=3)

# juglling with Pytorch versions and different module packaging
# this piece of code is probably a no-op
ckpt = torch.load(args.ckpt_dir, map_location="cpu")["g_ema"]
if "module" in list(ckpt.items())[0][0]:
    ckpt_adapt = OrderedDict()
    for k in ckpt.keys():
        k0 = k[7:]
        ckpt_adapt[k0] = ckpt[k]
    G.load_state_dict(ckpt_adapt)
else:
    G.load_state_dict(ckpt)
G.eval()
G = G.to(device)

print("G loaded")

# instatiating initial values for betas and alphas
if args.start == "ones":
    betas = torch.ones((14,), dtype=torch.float32, requires_grad=True, device=device)
    alphas = 0.5 * torch.ones((14,), dtype=torch.float32, device=device)
elif args.start == "zeros":
    betas = torch.zeros((14,), dtype=torch.float32, requires_grad=True, device=device)
    alphas = -1.0 * torch.ones((14,), dtype=torch.float32, device=device)
else:
    raise RuntimeError("Initial value unspecified")

alphas = alphas.requires_grad_()
optimizer = optim.Adam([alphas, betas], lr=args.lr0)
track = [[], [], [], []]

# "training" / calibrations loop
for epoch in range(args.n_epochs):
    shuffle(ensemble_dataset)
    print("#" * 80)
    print(f"Epoch {epoch}")
    print("#" * 80)
    pbar = tqdm(len(ensemble_dataset))
    for idx, (date, lt) in enumerate(ensemble_dataset):
        date_str = date.strftime('%Y-%m-%d')
        # loading inverted AROME members in latent space
        batch_w = torch.tensor(
            np.load(
                args.fake_data_dir + f"w_{date_str}_{lt}_{args.invstep}.npy"
            ).astype(np.float32)
        ).to(device)

        # loading reference (pre-normalized) ensemble data
        batch_y = torch.tensor(
            np.load(args.pack_dir + f"Rsemble_{date_str}_{lt}.npy").astype(
                np.float32
            )
        ).to(device)

        t = idx / len(ensemble_dataset)
        optim.lr = learning_rate(t, args.lr0)
        betas_noise = add_noise(betas, sigma(t, epoch), device)
        alphas_noise = add_noise(alphas, sigma(t, epoch), device)

        # using the K matrices : either loading it (to save time on later epochs)
        # or computing it on the fly
        try:
            K = torch.load(
                args.fake_data_dir
                + f"K_{date_str}_{lt}_{args.style_cut}_{args.invstep}.pt"
            )
            w_avg = torch.load(
                args.fake_data_dir
                + f"w_avg_{date_str}_{lt}_{args.style_cut}_{args.invstep}.pt"
            )

            if w_avg.shape != (args.style_cut, 512):
                K, w_avg = pca.compute_K_covariance(
                    batch_w[:, : args.style_cut], cut=args.n_samples - 1
                )
                torch.save(
                    K,
                    args.fake_data_dir
                    + f"K_{date_str}_{lt}_{args.style_cut}_{args.invstep}.pt",
                )
                torch.save(
                    w_avg,
                    args.fake_data_dir
                    + f"w_avg_{date_str}_{lt}_{args.style_cut}_{args.invstep}.pt",
                )

        except FileNotFoundError:

            # computation if not loaded
            K, w_avg = pca.compute_K_covariance(
                batch_w[:, : args.style_cut], cut=args.n_samples - 1
            )
            torch.save(
                K,
                args.fake_data_dir
                + f"K_{date_str}_{lt}_{args.style_cut}_{args.invstep}.pt",
            )
            torch.save(
                w_avg,
                args.fake_data_dir
                + f"w_avg_{date_str}_{lt}_{args.style_cut}_{args.invstep}.pt",
            )
        # sanity check
        try:
            assert w_avg.shape == (args.style_cut, 512)
        except AssertionError:
            print(date, lt, batch_w.shape, w_avg.shape)
            raise AssertionError("Uncorrect shape")

        # generating new 16 samples from seed
        gen = smpca.fast_style_mixing(
            alphas_noise,
            betas_noise,
            batch_w,
            K,
            w_avg,
            G,
            Whitening,
            device=device,
            beta_rule=args.beta_rule,
        )

        # if using rather wind speed than temperature
        if args.convert_ff_t:
            gen, batch_y = convert_uvt2fft(gen, batch_y)

        # loss on ensemble bias
        mean_loss = F.l1_loss(gen.mean(dim=0), batch_y.mean(dim=0))

        # spread inflation 1 + lambda
        lambda_inflation = (
            1.0 + args.inflate
            if not args.inflate_random
            else (1.0 + uniform(0, args.inflate))
        )

        # loss on ensemble spread
        std_loss = F.l1_loss(
            torch.std(gen, dim=0, unbiased=True),
            lambda_inflation * torch.std(batch_y, dim=0, unbiased=True),
        )

        # regressing
        loss = args.lambda_bias * mean_loss + args.lambda_spread * std_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keeping exponential moving averages to get smoother values (since batch size may be small)
        emaloss = 0.9 * emaloss + 0.1 * loss.item() if idx > 0 else loss.item()

        if args.beta_rule == "sigmoid":
            ema_betas = (
                0.1 * F.sigmoid(betas.detach().cpu()).numpy() + 0.9 * ema_betas
                if idx > 0
                else F.sigmoid(betas.detach().cpu()).numpy()
            )
        else:
            ema_betas = (
                0.1 * betas.detach().cpu().numpy() + 0.9 * ema_betas
                if idx > 0
                else betas.detach().cpu().numpy()
            )
        ema_alphas = (
            0.1 * F.sigmoid(alphas).detach().cpu().numpy() + 0.9 * ema_alphas
            if idx > 0
            else F.sigmoid(alphas).detach().cpu().numpy()
        )

        with torch.no_grad():
            emabias = (
                mean_loss.item() * 0.1 + 0.9 * emabias if idx > 0 else mean_loss.item()
            )

        pbar.set_description(
            f"t : {t:.3f}, ema loss (0.9) : {emaloss:.4f}, ema bias (0.9) : {emabias:.4f}, lr {learning_rate(t,args.lr0):.4f}, sigma : {sigma(t,epoch):.4f}"
        )

        # logged quantities
        track[0].append(emaloss)
        track[1].append(ema_betas)
        track[2].append(ema_alphas)
        track[3].append(emabias)

        if (idx % 512) == 0:

            print(f"betas : {betas.detach().cpu().numpy()}", flush=True)
            print(f"alphas : {F.sigmoid(alphas.detach().cpu()).numpy()}", flush=True)
        if (idx % 512) == 0 or idx == len(ensemble_dataset) - 1:
            # plotting verification : variances
            var_gen = torch.std(gen.detach(), dim=0, unbiased=True).cpu().numpy()
            var_real = torch.std(batch_y.detach(), dim=0, unbiased=True).cpu().numpy()

            n_var = 2 if args.convert_ff_t else 3
            fig, axs = plt.subplots(2, n_var, figsize=(6, 6), sharex=True, sharey=True)
            for j in range(n_var):
                cmap = "viridis" if (j < n_var - 1) else "coolwarm"
                axs[0, j].imshow(var_real[j], origin="lower", cmap=cmap)
                axs[1, j].imshow(
                    var_gen[j],
                    origin="lower",
                    cmap=cmap,
                    vmin=var_real[j].min(),
                    vmax=var_real[j].max(),
                )

            fig.tight_layout()
            plt.savefig(
                output_dir + f"std_real_vs_fake_{date}_{lt}_{idx}_{t:.3f}_{epoch}.png"
            )
            plt.close()

            # plotting verification : mean wind speed (in normalized space)
            if not args.convert_ff_t:
                with torch.no_grad():
                    gen, batch_y = convert_uvt2fft(gen.detach(), batch_y.detach())
            ff_gen = gen[:, 0].detach().cpu().mean(dim=0).numpy()
            ff_real = batch_y[:, 0].detach().cpu().mean(dim=0).numpy()
            fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)

            axs[0].imshow(ff_gen, origin="lower", cmap="viridis")
            axs[1].imshow(ff_real, origin="lower", cmap="viridis")

            fig.tight_layout()
            plt.savefig(
                output_dir
                + f"ff_mean_real_vs_fake_{date}_{lt}_{idx}_{t:.3f}_{epoch}.png"
            )
            plt.close()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].plot(track[0])
    for j in range(14):
        # plotting K- fluctuations different style than random fluctuations
        if j <= args.style_cut:
            axs[1].plot(np.array(track[1])[:, j])
            axs[2].plot(np.array(track[2])[:, j])
        else:
            axs[1].plot(np.array(track[1])[:, j], linestyle="dashed")
            axs[2].plot(np.array(track[2])[:, j], linestyle="dashed")
    axs[3].plot(track[3])
    axs[1].set_yscale("log")
    axs[2].set_yscale("log")
    fig.tight_layout()
    plt.savefig(output_dir + f"loss_betas_alphas_bias.png")
    plt.close()

    # saving losses and coefficients
    np.save(output_dir + "ema_loss.npy", track[0])
    np.save(output_dir + "ema_betas.npy", track[1])
    np.save(output_dir + "ema_alphas.npy", track[2])
    np.save(output_dir + "ema_bias.npy", track[3])
