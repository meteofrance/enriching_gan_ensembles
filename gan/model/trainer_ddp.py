#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:11:24 2023

@authors: brochetc rabaultj
"""

import time
from functools import wraps
from os import cpu_count
from time import perf_counter

import numpy as np
import torch

########## MP libs ############################################################
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import gan.data.dataset_handler_ddp as DSH
import gan.metrics4arome as METR
import gan.model.GAN_logic as GAN
import gan.plot.plotting_functions as plotFunc
from gan.distributed import get_rank, get_world_size, is_main_gpu, synchronize

########## NN libs ############################################################


########## data handling and plotting libs ####################################


# TODO : augmentation implementation
# TODO : conditional setup

###################### Scheduler choice function ##############################


def AllocScheduler(sched, optimizer, gamma, network):
    if sched == "exp":
        if is_main_gpu():
            print(f"{network} scheduler set to exponential scheduler")
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched == "linear":
        if is_main_gpu():
            print(f"{network} scheduler set to linear scheduler")
        lambda0 = lambda epoch: 1.0 / (1.0 + gamma * epoch)
        return optim.lr_scheduler.LambdaLR(optimizer, lambda0)
    else:
        if is_main_gpu():
            print(f"{network} scheduler set to None")
        return None


#################### some utils ####################################


def timer(func):
    @wraps(func)
    def time_measurement(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = perf_counter()
        res = func(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = perf_counter() - t0
        print(f"Function {func.__name__} Took {t1:.4f} seconds")
        return res

    return time_measurement


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def merge_dict(dic1, dic2):
    """
    fusion two dictionaries with different keys
    Warning if some keys are shared this will erase the values of dic1 and replace
    them with those of dic2
    """
    dicres = {k: v for d in (dic1, dic2) for k, v in d.items()}
    return dicres


def update_dict(dic1, dic2):
    """
    update dic1 list items with values from dic2 (with append)
    it is assumed that all values are torch.tensors
    """
    if len(dic1) == 0:
        for k, v in dic2.items():
            dic1[k] = [v.mean().item()]
    else:

        for k, v in dic2.items():
            dic1[k].append(v.mean().item())

    return dic1


###############################################################################
######################### Trainer class #######################################
###############################################################################


class Trainer:
    """
    main training class

    inputs :

        config -> model, training, optimizer, dirs, file names parameters
                (see constructors and main.py file)
        optimizer -> the type of optimizer (if working with LA-Minimax)

        test_metrics -> list of names from metrics namespace
        criterion -> specific metric name (in METR) used as save/stop criterion

        metric_verbose -> check if metric long name should be printed @ test time
                default to True

    outputs :

        models saved along with optimizers and schedulers at regular steps
        plots and scores logged regularly
        return of metrics and losses lists


    bound methods :

        "get-ready" methods:

            instantiate_optimizers : prepare distributed optimizers settings

            prepare_data_pipeline : create Dataset and Dataloader instances

            choose_algorithm : select GAN logic to implement

            instantiate_metric_log : prepare output file

            instantiate : Builder from the above methods + load and broadcast
                          models

        logging methods :
            log : save scores in file, including all losses

            plot : plot samples from generated and real distributions

            save_models : save models, optimizers, schedulers

            save_samples : save a large chunk of samples in a single array

        training methods :

            Discrim_Update

            Generator_Update

            test_ : provide models evaluation over variate metrics

            fit_ : train the models according to selected logic
                   pilot the testing
                   callbacks management
    """

    ############ timer decoration

    def __init__(self, config, criterion, test_metrics={}, metric_verbose=True):

        self.test_dataloader = None
        self.scheduler_D = None
        self.scheduler_G = None
        self.config = config
        self.instance_flag = False

        # self.test_metrics = []
        # for name in test_metrics:
        #    print("name ", name)
        #    print("getattr ", getattr(METR, name))
        #    print("type getattr ", type(getattr(METR, name)))
        #    self.test_metrics.append(getattr(METR, name))
        self.test_metrics = [getattr(METR, name) for name in test_metrics]
        # print("test_met ", test_metrics)
        # print("self.test_met ", self.test_metrics)
        self.criterion = getattr(METR, criterion)

        self.metric_verbose = metric_verbose
        self.batch_size = self.config.batch_size

        self.test_step = self.config.test_step
        self.save_step = self.config.save_step
        self.plot_step = self.config.plot_step
        self.test_epoch = self.config.test_epoch
        self.save_epoch = self.config.save_epoch
        self.plot_epoch = self.config.plot_epoch

        self.crop_indexes = self.config.crop_indexes

        self.loss_dict = {}

        self.Metrics = {name: [] for metr in self.test_metrics for name in metr.names}
        self.crit_List = []

    ########################## GETTING-READY FUNCTIONS ########################

    def instantiate_optimizers(self, modelG, modelD, load_optim):
        """
        instantiate and prepare optimizers and lr schedulers
        """

        g_reg_ratio = self.config.g_reg_every / (self.config.g_reg_every + 1)
        d_reg_ratio = self.config.d_reg_every / (self.config.d_reg_every + 1)

        self.optim_G = optim.Adam(
            modelG.parameters(),
            lr=self.config.lr_G * g_reg_ratio,
            betas=(
                self.config.beta1_G**g_reg_ratio,
                self.config.beta2_G**g_reg_ratio,
            ),
        )

        self.optim_D = optim.Adam(
            modelD.parameters(),
            lr=self.config.lr_D * d_reg_ratio,
            betas=(
                self.config.beta1_D**d_reg_ratio,
                self.config.beta2_D**g_reg_ratio,
            ),
        )

        if load_optim is not None:
            self.optim_G.load_state_dict(load_optim["g_optim"])
            self.optim_D.load_state_dict(load_optim["d_optim"])

        if self.config.scheduler_config != "None":
            print(f"Scheduler: {self.config.config_dir}{self.config.scheduler_config}")
            with open(
                f"{self.config.config_dir}{self.config.scheduler_config}", "r"
            ) as sched_config_file:
                sched_yaml = yaml.safe_load(sched_config_file)
            self.scheduler_G = AllocScheduler(
                sched_yaml["G_sched"],
                self.optim_G,
                sched_yaml["lrG_gamma"],
                "Generator",
            )
            self.scheduler_D = AllocScheduler(
                sched_yaml["D_sched"],
                self.optim_D,
                sched_yaml["lrD_gamma"],
                "Discriminator",
            )
        else:
            print(f"No scheduler_config file provided, shedulers set to None")
            if self.config.scheduler_config != "" and self.config.pretrained_model > 0:
                print("Loading scheduler...")
                self.scheduler_G.load_state_dict(
                    torch.load(
                        f"{self.config.output_dir}/models/SchedGen_{self.config.pretrained_model}"
                    )
                )

                self.scheduler_D.load_state_dict(
                    torch.load(
                        f"{self.config.output_dir}/models/SchedDisc_{self.config.pretrained_model}"
                    )
                )

        return torch.cuda.memory_allocated()

    def prepare_data_pipeline(self):
        """

        instantiate datasets and dataloaders to be used in training
        set GPU-CPU communication parameters
        see ISData_Loader classes for details

        """
        torch.set_num_threads(1)

        if is_main_gpu():
            print("Loading data")
        self.Dl_train = DSH.ISData_Loader("Train", self.config)

        self.Dl_test = DSH.ISData_Loader("Test", self.config)

        kwargs = {"pin_memory": True}

        self.train_dataloader = self.Dl_train.loader(
            get_world_size(), get_rank(), kwargs
        )
        self.test_dataloader = self.Dl_test.loader(get_world_size(), get_rank(), kwargs)

    def choose_algorithm(self):

        if self.config.train_type in ["stylegan", "wave_d"]:
            self.D_backward = GAN.Discrim_Step_StyleGAN
            self.G_backward = GAN.Generator_Step_StyleGAN

    def instantiate_metrics_log(self):
        """create metrics to be logged
        prepare log file
        # metrics list is absolutely agnostic, so a rough list of metric functions
        # as basis for test_metrics seems good practice here"""

        data_names = ["Step"]
        data_names += [k for k in self.loss_dict.keys()]
        data_names += ["criterion"]

        if len(self.Metrics) > 0:
            data_names += [name for name in self.Metrics.keys()]

        mode = "w" if self.config.pretrained_model == 0 else "a"
        with open(self.config.output_dir + "/log/metrics.csv", mode) as file:

            for i, mname in enumerate(data_names):
                if i == 0:
                    file.write(mname)
                else:
                    file.write("," + mname)
            file.write("\n")
            file.close()

    def instantiate(self, modelG, modelD, load_optim=False, modelG_ema=None):

        torch.manual_seed(self.config.seed)
        mem_cuda = torch.cuda.memory_allocated()
        torch.manual_seed(self.config.seed)
        modelD.cuda()
        mem_d = torch.cuda.memory_allocated() - mem_cuda
        modelG.cuda()
        mem_g = torch.cuda.memory_allocated() - mem_d - mem_cuda

        if modelG_ema is not None:
            modelG_ema.cuda()
            modelG_ema = DDP(
                modelG_ema,
                device_ids=[get_rank()],
                output_device=get_rank(),
                broadcast_buffers=False,
            )
            # modelG_ema = torch.compile(modelG_ema)

        modelG = DDP(
            modelG,
            device_ids=[get_rank()],
            output_device=get_rank(),
            broadcast_buffers=False,
        )
        # modelG = torch.compile(modelG)
        modelD = DDP(
            modelD,
            device_ids=[get_rank()],
            output_device=get_rank(),
            broadcast_buffers=False,
        )
        # modelD = torch.compile(modelD)
        ###

        mem_opt = self.instantiate_optimizers(modelG, modelD, load_optim)

        self.choose_algorithm()

        self.prepare_data_pipeline()

        #######################################################################

        if is_main_gpu():
            print("Data loaded, Trainer instantiated")
        self.instance_flag = True

        return tuple(
            v
            for v in [modelG, modelD, modelG_ema, mem_g, mem_d, mem_opt, mem_cuda]
            if v is not None
        )

    ################################ LOGGING FUNCTIONS #######################

    def log(self, Step):

        data = [Step]
        data_to_write = "%d"

        for lname in self.loss_dict.keys():
            data_to_write = data_to_write + ",%.6f"
            data += [self.loss_dict[lname][-1]]

        data += [self.crit_List[-1]]
        data_to_write = data_to_write + ",%.6f"

        for mname in self.Metrics.keys():
            data_to_write = data_to_write + ",%.6f"
            data += [self.Metrics[mname][-1]]

        data_to_write = data_to_write % tuple(data)

        with open(self.config.output_dir + "/log/metrics.csv", "a") as file:
            file.write(data_to_write)
            file.write("\n")
            file.close()

    def plot(self, Step, modelG, samples, samples_z):
        print("Plotting")

        modelG.eval()

        with torch.no_grad():
            batch, _, _ = modelG([samples_z])

        modelG.train()

        batch = torch.cat(
            (batch, samples[: self.config.plot_samples // 4, :, :, :]), dim=0
        )

        plotFunc.online_sample_plot(self.config, batch, Step)

    def save_models(self, Step, modelG, modelD, modelG_ema=None):
        print("Saving Models")

        # models
        torch.save(
            {
                "g": modelG.state_dict(),
                "d": modelD.state_dict(),
                "g_ema": modelG_ema.state_dict(),
                "g_optim": self.optim_G.state_dict(),
                "d_optim": self.optim_D.state_dict(),
            },
            self.config.output_dir + f"/models/{str(Step).zfill(6)}.pt",
        )

        # schedulers
        if self.scheduler_D is not None:
            torch.save(
                self.scheduler_D.state_dict(),
                self.config.output_dir + "/models/SchedDisc_{}".format(Step),
            )
        if self.scheduler_G is not None:
            torch.save(
                self.scheduler_G.state_dict(),
                self.config.output_dir + "/models/SchedGen_{}".format(Step),
            )

    def save_samples(self, number, Step, modelG):
        print("Saving samples")
        # Moving back and forth the model to cpu and cuda is absolutely NOT optimal but seems a
        # good solution to prevent cuda running out of memory
        # modelG.cpu()
        for i in range(16):
            z = torch.empty(number, self.config.latent_dim).normal_().cuda()
            # z = torch.empty(number,self.config.latent_dim).normal_().cpu()
            with torch.no_grad():
                out = modelG([z])[0].cpu().numpy()

            np.save(
                self.config.output_dir + "/samples/_Fsample_{}_{}.npy".format(Step, i),
                out,
            )
        # modelG.cuda()
        return 0

    ########################### TRAINING FUNCTIONS ############################

    def Discrim_Update(self, modelD, modelG, samples, step=0):

        requires_grad(modelG, False)
        requires_grad(modelD, True)

        samples = samples.cuda()

        loss_0 = self.D_backward(
            samples,
            modelD,
            modelG,
            self.config.latent_dim,
            mixing=self.config.mixing,
        )

        self.optim_D.step()

        if self.config.train_type == "stylegan" and step % self.config.d_reg_every == 0:
            loss_1 = GAN.Discrim_Regularize(
                samples, modelD, self.config.r1, self.config.d_reg_every
            )

            loss_0 = merge_dict(loss_0, loss_1)
            self.optim_D.step()
        # self.optim_D.step()
        return loss_0, samples

    def Generator_Update(self, modelD, modelG, samples, step=0):

        requires_grad(modelG, True)
        requires_grad(modelD, False)

        loss_0 = self.G_backward(
            samples,
            modelD,
            modelG,
            self.config.latent_dim,
            mixing=self.config.mixing,
        )

        self.optim_G.step()

        if self.config.train_type == "stylegan" and step % self.config.g_reg_every == 0:
            requires_grad(modelG, True)
            requires_grad(modelD, False)

            path_batch_size = max(
                1, self.config.batch_size // self.config.path_batch_shrink
            )

            loss_1, mean_path_length = GAN.Generator_Regularize(
                path_batch_size,
                self.config.path_regularize,
                self.mean_path_length,
                modelG,
                self.config.latent_dim,
                self.config.g_reg_every,
                self.config.path_batch_shrink,
                mixing=self.config.mixing,
            )

            loss_0 = merge_dict(loss_0, loss_1)
            self.optim_G.step()

        elif self.config.train_type == "wave_d" and step % self.config.g_reg_every == 0:
            requires_grad(modelG, True)
            requires_grad(modelD, False)

            path_batch_size = max(
                1, self.config.batch_size // self.config.path_batch_shrink
            )

            loss_1, mean_path_length = GAN.Generator_Regularize(
                path_batch_size,
                self.config.path_regularize,
                self.mean_path_length,
                modelG,
                self.config.latent_dim,
                self.config.g_reg_every,
                self.config.path_batch_shrink,
                mixing=self.config.mixing,
            )

            loss_0 = merge_dict(loss_0, loss_1)
            self.optim_G.step()

        return loss_0

    def test_(self, modelG, DataIter):
        """
        test samples in parallel for each metric
        iterates through test dataset with DataIter
        """

        modelG.eval()
        real_samples, _, _ = next(DataIter)
        real_samples = real_samples.cuda()
        sample_num = min(real_samples.shape[0], self.config.test_samples)

        # using sample_num samples PER GPU to compute scores

        z = torch.empty((sample_num, self.config.latent_dim)).normal_().cuda()

        with torch.no_grad():
            fake_samples, _, _ = modelG([z])
        metric_results = {}

        for metr in self.test_metrics:
            res = metr(real_samples, fake_samples, select=False)
            assert torch.isfinite(res).all()

            if is_main_gpu():
                for i, name in enumerate(metr.names):
                    self.Metrics[name].append(res[i].item())
                metric_results[metr.long_name] = res

        if is_main_gpu():
            print(
                "\t Instances  "
                + "  ".join(
                    [name.ljust(9) for metr in self.test_metrics for name in metr.names]
                )
            )
            print(f"\t {len(real_samples):<9}", end="")
            for metr in self.test_metrics:
                for val in metric_results[metr.long_name]:
                    print(f"{   val.item():.4f}".ljust(9), end="")
            print()

        crit = self.criterion(real_samples, fake_samples, select=False)

        assert torch.isfinite(crit).all()

        if is_main_gpu():
            self.crit_List.append(crit.item())

    def fit_(self, modelG, modelD, modelG_ema=None):

        assert (
            self.instance_flag
        )  # security flag avoiding meddling with uncomplete init

        samples_z = (
            torch.empty(3 * (self.config.plot_samples // 4), self.config.latent_dim)
            .normal_()
            .cuda()
        )

        ### generator EMA will be evaluated on samples_z for plotting

        if modelG_ema is not None:
            G_module = modelG

        synchronize()
        Step = 0

        self.mean_path_length = torch.tensor([0.0], dtype=torch.float32)

        accum = 0.5 ** (32 / (10 * 1000))  # EMA parameter

        N_batch = len(self.train_dataloader)

        if is_main_gpu():
            print(
                "Starting training for", self.config.epochs_num, "epochs...", flush=True
            )

        for epoch in range(self.config.epochs_num):

            self.train_dataloader.sampler.set_epoch(epoch)
            self.test_dataloader.sampler.set_epoch(epoch)
            if is_main_gpu():
                print(
                    "----------------------------------------------------------",
                    flush=True,
                )
                print("Epoch num ", epoch + 1, "/", self.config.epochs_num, flush=True)

            step = 0
            time_step = 0.0
            step_tmp = 0

            if is_main_gpu():
                loop = tqdm(
                    enumerate(self.train_dataloader),
                    total=len(self.train_dataloader),
                    desc=f"Epoch {epoch + 1}/{self.config.epochs_num}",
                    unit="batch",
                    postfix=f"",
                )
            else:
                loop = enumerate(self.train_dataloader)

            for i, batch in loop:
                img, _, _ = batch
                t = time.perf_counter()
                Step = epoch * N_batch + step
                if self.config.pretrained_model > 0:
                    Step += self.config.pretrained_model

                true_epoch = (
                    epoch + self.config.pretrained_model
                )  # used only in output funcs

                ############################### Discriminator Updates #########
                loss_d, samples = self.Discrim_Update(
                    modelD, modelG, img, step=Step
                )  # all losses are already reduced after this step

                ############################ Generator Update #################

                loss_g = self.Generator_Update(
                    modelD, modelG, samples, step=Step
                )  # all losses are already reduced after this step

                if is_main_gpu():
                    loop.set_postfix_str(
                        f"loss_d : {loss_d['d'].item():.5f} | loss_g : {loss_g['g'].item():.5f} | step: {step:.2f}"
                    )
                    # print("up")

                ################ advancing schedulers for learning rate #######
                if step == N_batch - 1:
                    if self.scheduler_G is not None:
                        self.scheduler_G.step()
                    if self.scheduler_D is not None:
                        self.scheduler_D.step()

                time_step += time.perf_counter() - t

                ############################ Performing EMA step ##############

                if modelG_ema is not None:
                    accumulate(modelG_ema, G_module, accum)

                ########################### Update training metrics
                losses = merge_dict(loss_d, loss_g)

                self.loss_dict = update_dict(self.loss_dict, losses)

                step += 1
                step_tmp += 1

                if (
                    self.save_step > 0
                    and (
                        Step % self.save_step == 0
                        or Step == self.config.total_steps - 1
                    )
                    and is_main_gpu()
                ):
                    self.save_samples(self.config.sample_num, Step, modelG_ema)
                    self.plot(Step, modelG_ema, samples, samples_z)
                    if (
                        self.config.pretrained_model > 0
                    ):  # we don't want to save the first step when using a pretrained
                        if not (Step == self.config.pretrained_model):
                            self.save_models(Step, modelG, modelD, modelG_ema)
                    else:
                        self.save_models(Step, modelG, modelD, modelG_ema)

                if (
                    Step > self.config.total_steps
                    and self.config.total_steps / N_batch < self.config.epochs_num
                ):
                    break
                ############ END OF STEP ######################################

            if epoch == 0 and is_main_gpu():
                self.instantiate_metrics_log()

            ############### saving models and samples #################

            if (
                self.save_epoch > 0
                and (
                    epoch % self.save_epoch == 0 or epoch == self.config.epochs_num - 1
                )
                and is_main_gpu()
            ):
                self.save_samples(self.config.sample_num, true_epoch, modelG_ema)
                if (
                    self.config.pretrained_model > 0
                ):  # we don't want to save the first step when using a pretrained
                    if not (true_epoch == self.config.pretrained_model and epoch == 0):
                        self.save_models(true_epoch, modelG, modelD, modelG_ema)
                else:
                    self.save_models(true_epoch, modelG, modelD, modelG_ema)

            ############### testing samples at test epoch ##################

            if self.test_epoch > 0 and (
                epoch % self.test_epoch == 0 or epoch == self.config.epochs_num - 1
            ):
                test_Dl_iter = iter(self.test_dataloader)
                self.test_(modelG_ema, test_Dl_iter)

            ############# logging experiment data #############

            if (
                self.config.log_epoch > 0
                and (
                    epoch % self.config.log_epoch == 0
                    or epoch == self.config.epochs_num - 1
                )
                and is_main_gpu()
            ):
                self.log(true_epoch)

            ############### plotting distribution at plot epoch ############
            if (
                self.plot_epoch > 0
                and (
                    epoch % self.plot_epoch == 0 or epoch == self.config.epochs_num - 1
                )
                and is_main_gpu()
            ):
                self.plot(true_epoch, modelG_ema, samples, samples_z)

            if self.config.total_steps > 1000:
                if step_tmp % 1000 == 0 and step_tmp != 0 and is_main_gpu():
                    print(
                        f"Time taken: about {time_step/step_tmp}s per step (not counting logs and metrics)."
                    )
                    time_step = 0
                    step_tmp = 0
            if self.config.total_steps > 200 and self.config.total_steps <= 1000:
                if step_tmp % 50 == 0 and step_tmp != 0 and is_main_gpu():
                    print(
                        f"Time taken: about {time_step/step_tmp}s per step (not counting logs and metrics)."
                    )
                    time_step = 0
                    step_tmp = 0
            if self.config.total_steps <= 200 and is_main_gpu():
                if step_tmp % 10 == 0 and step_tmp != 0:
                    print(
                        f"Time taken: about {time_step/step_tmp}s per step (not counting logs and metrics)."
                    )
                    time_step = 0
                    step_tmp = 0

        ################ END OF EPOCH #####################################

        ##########################
        return self.loss_dict, self.crit_List, self.Metrics


###############################################################################
