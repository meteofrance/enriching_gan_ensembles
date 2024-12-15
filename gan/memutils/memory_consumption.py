import pandas as pd
import torch
import torch.cuda as cu
import torch.optim as optim

from gan.distributed import is_main_gpu


def mem_stylegan2(input_dim, bs, modelG, modelD, mem_cuda=0, multi_gpu_opt="horovod"):
    optim_D = optim.Adam(modelD.parameters(), lr=0.002)
    optim_G = optim.Adam(modelG.parameters(), lr=0.002)
    mem_model = mem_cuda

    if multi_gpu_opt == "horovod" or multi_gpu_opt == "ddp":

        Z = torch.randn(bs, input_dim).cuda()

        pred_X = modelG([Z])[0]
        mem1 = cu.memory_allocated()

        pred_Z = modelD(pred_X)

        loss = -pred_Z.mean()

        mem2 = cu.memory_allocated()

        optim_G.zero_grad()
        optim_D.zero_grad()
        loss.backward()
        optim_G.step()

        optim_D.step()
        mem3 = cu.memory_allocated()

        # cu.memory_summary()
        ###freeing memory ###
        del loss, Z, pred_X, pred_Z, optim_G, optim_D
        torch.cuda.empty_cache()
        # cu.memory_summary()
    else:
        raise ValueError("Unknown multi gpu framework {}".format(multi_gpu_opt))

    return (max(mem1, max(mem2, mem3)) - mem_model) / 1024**3


def log_mem_consumption(modelG, modelD, config, mem_g, mem_d, mem_opt, mem_cuda):
    params_gen = [p.numel() for p in modelG.parameters() if p.requires_grad]
    params_dis = [p.numel() for p in modelD.parameters() if p.requires_grad]
    Gen_param = sum(params_gen)
    Dis_param = sum(params_dis)

    print(Gen_param, Dis_param)

    df = pd.DataFrame(
        data=0,
        index=[
            "Cuda_inst",
            "Generator",
            "Discriminator",
            "Optim",
            "Batch_max",
            "Total",
        ],
        columns=["Nb_elmt", "Memory (GiB)", "bs", "size", "nb_var"],
    )

    mem_batch = 0  # mem_stylegan2(config.latent_dim, config.batch_size,
    #         modelD=modelD, modelG=modelG, mem_cuda=mem_cuda)

    df.loc["Cuda_inst"] = [
        0,
        mem_cuda / 1024**3,
        config.batch_size,
        config.crop_size[0],
        len(config.var_names),
    ]  # if not config.mean_pert else len(config.var_names)*2]
    df.loc["Generator"] = [Gen_param, mem_g / 1024**3, 0, 0, 0]
    df.loc["Discriminator"] = [Dis_param, mem_d / 1024**3, 0, 0, 0]
    df.loc["Optim"] = [0, mem_opt / 1024**3, 0, 0, 0]
    df.loc["Batch_max"] = [0, mem_batch, 0, 0, 0]
    df.loc["Total"] = [sum(df["Nb_elmt"]), sum(df["Memory (GiB)"]), 0, 0, 0]
    if is_main_gpu():
        print(df)

    if config.pretrained_model == 0:
        df.to_csv(config.output_dir + "/df_size_param.csv")
