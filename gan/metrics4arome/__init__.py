#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:53:46 2022

@author: brochetc

metrics version 2

File include :
    
    metric2D and criterion2D APIs to be used by Trainer class from trainer_horovod
    provide a directly usable namespace from already implemented metrics

"""
import gan.metrics4arome.general_metrics as GM
import gan.metrics4arome.length_scales as ls

# import metrics4arome.inception_metrics as inception
# import metrics4arome.scattering_metric as scat
# import metrics4arome.structure_functions as sfunc
import gan.metrics4arome.multivariate as multiv
import gan.metrics4arome.quantiles_metric as quant
import gan.metrics4arome.sliced_wasserstein as SWD
import gan.metrics4arome.spectrum_analysis as Spectral
import gan.metrics4arome.wasserstein_distances as WD

###################### standard parameters

var_dict = {
    "rr": 0,
    "u": 1,
    "v": 2,
    "t2m": 3,
    "orog": 4,
    "z500": 5,
    "t850": 6,
    "tpw850": 7,
}

all_var = list(var_dict.keys())
all_var_no_rr_orog = ["u", "v", "t2m", "z500", "t850", "tpw850"]
vars_u_v_t = ["u", "v", "t2m"]
vars_u_v = ["u", "v"]
vars_t = ["t2m"]
vars_rr = ["rr"]
vars_rr_u_v = ["rr", "u", "v"]
vars_rr_t2m = ["rr", "t2m"]
vars_rr_u_v_t2m = ["rr", "u", "v", "t2m"]


######################

########################### High level APIs ##################################


class metric2D:
    def __init__(self, long_name, func, variables, names=["metric"], mean_pert=False):

        self.long_name = long_name

        self.names = names  # names for each of the func's output items

        self.func = (
            func  # should return np.array OR tensor to benefit from parallel estimation
        )

        self.variables = variables  # variables on which the metric is applied

        self.mean_pert = mean_pert

    def selectVars(self, *args):
        """
        select in the input data the variables to compute metric on
        """

        if len(args[0]) == 2:

            real_data, fake_data = args[0]
            var_dict_f = {
                var: i
                for i, var in enumerate(
                    sorted(self.variables, key=lambda var_f: var_dict[var_f])
                )
            }
            if self.mean_pert:
                var_dict_f.update(
                    {
                        var + "_mean": i + 8
                        for i, var in enumerate(
                            sorted(self.variables, key=lambda var_f: var_dict[var_f])
                        )
                    }
                )
                var_dict.update(
                    {
                        var + "_mean": i + 8
                        for i, var in enumerate(
                            sorted(all_var, key=lambda var_f: var_dict[var_f])
                        )
                    }
                )

            VI = [var_dict[v] for v in self.variables]
            VI_f = [
                var_dict_f[v] for v in self.variables
            ]  # changed here because otherwise indexes may be wrong for monovar
            if self.mean_pert:
                VI = VI + [var_dict[v + "_mean"] for v in self.variables]
                VI_f = VI_f + [var_dict_f[v + "_mean"] for v in self.variables]
            real_data = real_data[:, VI, :, :]
            fake_data = fake_data[:, VI_f, :, :]

            return real_data, fake_data

        else:

            return args[0]

    def __call__(self, *args, **kwargs):

        ########## selecting variables check #########
        try:
            select = kwargs["select"]
        except KeyError:

            select = True

        ############# selection ################

        if select:

            data = self.selectVars(args)

        else:

            data = args

        ########### computation ################

        reliq_kwargs = {k: v for k, v in kwargs.items() if k != "select"}

        if len(data) == 2:

            return self.func(data[0], data[1], **reliq_kwargs)

        else:

            return self.func(data[0], **reliq_kwargs)


#################
#################


class criterion2D(metric2D):
    def __init__(self, long_name, func, variables, names):
        super().__init__(long_name, func, variables, names)


##############################################################################
################## Metrics catalogue #####################

standalone_metrics = {
    "spectral_compute",
    "spectral_distrib",
    "struct_metric",
    "ls_metric",
    "ls_metric_all",
    "IntraMapVariance",
    "InterMapVariance",
}

distance_metrics = {
    "Orography_RMSE",
    "W1_Center",
    "W1_Center_NUMPY",
    "W1_random",
    "W1_random_NUMPY",
    "W1_center",
    "W1_Random",  # added center and Random for the mean/pert dataset (beware of caps)
    "pw_W1",
    "pw_W1_all",
    "spectral_dist",
    "ls_dist",
    "SWD_metric",
    "SWD_metric_torch",
    "SWD_metric_u",
    "SWD_metric_v",
    "SWD_metric_t2m",
    "SWD_metric_z500",
    "SWD_metric_t850",
    "SWD_metric_tpw850",  # those last 6 are for swd decoupled calcuations
    "fid",
    "scat_SWD_metric",
    "scat_SWD_metric_renorm",
    "multivar",
    "quant_metric",
    "spectral_dist_all",
    "spectral_dist_torch_all",
    "spectral_dist",
    "spectral_dist_rr",
    "spectral_dist_torch_rr",
    "spectral_dist_torch_rr_u_v",
    "spectral_dist_torch_rr_t2m",
    "spectral_dist_torch_rr_u_v_t2m",
}


###################### Usable namespace #######################################

Orography_RMSE = metric2D(
    "RMS Error on orography synthesis  ", GM.orography_RMSE, "orog"
)

# a changer pour suivre le modele de specral dist
IntraMapVariance = {
    "u_v_t2m": metric2D(
        "Mean intra-map variance of channels   ",
        GM.intra_map_var,
        vars_u_v_t,
        names=["intra_u", "intra_v", "intra_t2m"],
    ),
    "u_v": metric2D(
        "Mean intra-map variance of channels   ",
        GM.intra_map_var,
        vars_u_v_t,
        names=["intra_u", "intra_v"],
    ),
    "t2m": metric2D(
        "Mean intra-map variance of channels   ",
        GM.intra_map_var,
        vars_u_v_t,
        names=["intra_t2m"],
    ),
}

InterMapVariance = metric2D(
    "Mean Batch variance of channels   ",
    GM.inter_map_var,
    vars_u_v_t,
    names=["inter_u", "inter_v", "inter_t2m"],
)

## crude Wasserstein distances

W1_Center = criterion2D(
    "Mean Wasserstein distance on center crop  ",
    WD.W1_center,
    all_var_no_rr_orog,
    names=["W1_Center"],
)


W1_Center_NUMPY = criterion2D(
    "Mean Wasserstein distance on center crop  ",
    WD.W1_center_numpy,
    all_var_no_rr_orog,
    names=["W1_Center"],
)

W1_random = criterion2D(
    "Mean Wasserstein distance on random selection  ",
    WD.W1_random,
    all_var_no_rr_orog,
    names=["W1_random"],
)

W1_random_NUMPY = criterion2D(
    "Mean Wasserstein distance on random selection  ",
    WD.W1_random_NUMPY,
    all_var_no_rr_orog,
    names=["W1_random"],
)

pw_W1_all = metric2D(
    "Point Wise Wasserstein distance",
    WD.pointwise_W1,
    all_var_no_rr_orog,
    names=["pw_W1"],
)


# Sliced Wasserstein Distance estimations

sliced_w1 = SWD.SWD_API(image_shape=(128, 128), numpy=True)

SWD_metric_all = metric2D(
    "Sliced Wasserstein Distance  ",
    sliced_w1.End2End,
    all_var_no_rr_orog,
    names=sliced_w1.get_metric_names(),
)

sliced_w1_torch = SWD.SWD_API(image_shape=(128, 128), numpy=False)

SWD_metric_torch_all = metric2D(
    "Sliced Wasserstein Distance  ",
    sliced_w1_torch.End2End,
    all_var_no_rr_orog,
    names=sliced_w1_torch.get_metric_names(),
)


# spectral analysis


spectral_dist_all = metric2D(
    "Power Spectral Density RMSE  ",
    Spectral.PSD_compare,
    all_var_no_rr_orog,
    names=["PSDu", "PSDv", "PSDt2m", "PSDz500", "PSDt850", "PSDtpw850"],
)
spectral_dist_rr = metric2D(
    "Power Spectral Density RMSE  ", Spectral.PSD_compare, vars_rr, names=["PSDrr"]
)
spectral_dist_torch_rr = metric2D(
    "Power Spectral Density RMSE  ",
    Spectral.PSD_compare_torch,
    vars_rr,
    names=["PSDrr"],
)
spectral_dist_torch_all = metric2D(
    "Power Spectral Density RMSE  ",
    Spectral.PSD_compare_torch,
    all_var_no_rr_orog,
    names=["PSDu", "PSDv", "PSDt2m", "PSDz500", "PSDt850", "PSDtpw850"],
)
spectral_dist_torch_rr_u_v = metric2D(
    "Power Spectral Density RMSE  ",
    Spectral.PSD_compare_torch,
    vars_rr_u_v,
    names=["PSDrr", "PSDu", "PSDv"],
)
spectral_dist_torch_rr_t2m = metric2D(
    "Power Spectral Density RMSE  ",
    Spectral.PSD_compare_torch,
    vars_rr_t2m,
    names=["PSDrr", "PSDt2m"],
)
spectral_dist_torch_rr_u_v_t2m = metric2D(
    "Power Spectral Density RMSE  ",
    Spectral.PSD_compare_torch,
    vars_rr_u_v_t2m,
    names=["PSDrr", "PDSu", "PSDv", "PSDt2m"],
)

spectral_compute = metric2D(
    "Power Spectral Density  ", Spectral.PowerSpectralDensity, all_var
)

# spectral_distrib = metric2D('Power Spectral Density Distribution ', \
#                   Spectral.PowerSpectralDensity_Distrib, vars_wo_orog, names = ['PSDu', 'PSDv','PSDt2m']     )


# FID score
"""
fid = metric2D('Fréchet Inception Distance  ',\
             inception.FIDclass(inception.inceptionPath).FID,\
             ['FID'])

# scattering metrics with sparsity and shape estimators


scat_sparse = scat.scattering_metric(
        J=4,L=8,shape=(127,127), estimators=['s21', 's22'],
        frontend='torch', backend='torch', cuda=True
                                   )
#two versions of the same metrics
scat_SWD_metric = metric2D('Scattering Estimators ', scat_sparse.scattering_sliced,\
                       vars_wo_orog)

scat_SWD_metric_renorm = metric2D('Scattering Estimators', scat_sparse.scattering_renorm,
                              vars_wo_orog)

scat_rmse = metric2D('Scattering Estimators', scat_sparse.scattering_rmse,
                              vars_wo_orog)

# structure functions 

struct_metric = metric2D('First order structure function', 
                         lambda data : sfunc.increments(data, max_length = 16),\
                       vars_wo_orog)
"""
# multivariate_comparisons
multivar = metric2D(
    "Multivariate data ",
    multiv.multi_variate_correlations,
    all_var_no_rr_orog,
    names=["Corr_r", "Corr_f"],
)

# Correlation length maps

scale = 2.5  # number of kilometers per grid point
ls_metric_all = metric2D(
    "Correlation length maps",
    lambda data: ls.length_scale(data, sca=scale),
    all_var_no_rr_orog,
)

ls_dist = metric2D(
    "Correlation length RMSE",
    lambda real, fake: ls.length_scale_abs(real, fake, sca=scale),
    vars_u_v_t,
    names=["Lcorr_u", "Lcorr_v", "Lcorr_t2m"],
)

# quantile scores

qlist = [0.01, 0.1, 0.9, 0.99]

quant_metric = metric2D(
    "Quantiles RMSE score",
    lambda real, fake: quant.quantile_score(real, fake, qlist=qlist),
    vars_u_v_t,
)
