#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:13:47 2022

@author: brochetc and mouniera

artistic stuff with matplotlib & cartopy
To plot nice visualizations of GAN _generated or AI generated fields
"""


import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.patches as patches
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

cmapRR = colors.ListedColormap(
    [
        "white",
        "mediumpurple",
        "blue",
        "dodgerblue",
        "darkseagreen",
        "seagreen",
        "greenyellow",
        "yellow",
        "navajowhite",
        "sandybrown",
        "darkorange",
        "red",
        "darkred",
        "black",
    ],
    name="from_list",
    N=None,
)


def extract_lonlat(fpath="/home/mrmn/brochetc/"):
    # Extraction des longitudes,latitudes des pdg. A modifier si grille modifiée !

    filename = fpath + "styleganPNRIA/gan/projection_artistic/latlon.file"
    with open(filename, "rb") as f:
        lonlat = pickle.load(f, encoding="latin1")
    return lonlat


def grid_to_lat_lon(X, Y):
    """
    renormalize a pixel-wise grid to Latitude/longitude Coordinates

    Constants are fixed to center the domain on AROME domain

    """
    Lat_min = 37.5
    Lat_max = 55.4
    Lon_min = -12.0
    Lon_max = 16.0
    n_lat = 717
    n_lon = 1121
    Lat = Lat_min + Y * (Lat_max - Lat_min) / n_lat
    Lon = Lon_min + X * (Lon_max - Lon_min) / n_lon
    return Lat, Lon


def get_boundaries(Zone):
    """
    retrieves left-most, bottom-most domain boundaries for distinct climatologic regions
    over AROME-FRANCE

    Inputs :
        Zone : str the region to be selected

    Returns :
        X_min, Y_min : int; the indexes on AROME grid of the bottom left corner of the region

    """
    Zone_l = [
        "NO",
        "SO",
        "SE",
        "NE",
        "C",
        "SE_for_GAN",
        "SE_GAN_extend",
        "SE_for_GAN_terrestrial",
        "AROME_all",
    ]
    X_min = [230, 250, 540, 460, 360, 540 + 55, 540, 500, 0]
    Y_min = [300, 150, 120, 300, 220, 120 + 78, 120, 180, 0]
    index = Zone_l.index(Zone)
    return X_min[index], Y_min[index]


def cartesian2polar(u, v):
    """

    Transform cartesian notation (u,v) of a planar vector
    into its module+direction notation

    -------------
    Inputs :

        u, v, respectively the x- and y- coords of the vector

    Returns :
        module, direction :tuple

        The module and normalized coordinates (respectively cos(theta), sin(theta) in the polar framework)

    -------------
    """

    module = np.sqrt(u**2 + v**2)
    direction = (u / module, v / module)
    return module, direction


def standardize_samples(
    data,
    normalize=False,
    norm_vectors=(0.0, 1.0),
    chan_ind=None,
    ref_chan_ind=None,
    crop_inds=None,
):
    """
    --------------------------------
    Inputs :

        normalize : bool (optional) -> list of samples to be normalized

        norm_vectors : tuple(array) (optional) -> target mean and absolute maximum
                    to be used for normalization
                    array should be of size len(var_names)

        crop : bool (optional) -> check if data should be cropped

        crop_inds : list(int) -> crop indexes
                                (xmin, xmax, ymin, ymax)"""

    print(data.shape)
    if normalize:
        print("Normalizing")
        Means = norm_vectors[0]
        Maxs = norm_vectors[1]
    if chan_ind == None and crop_inds == None:

        print("Data is data clean")
        data_clean = data

    elif chan_ind is not None and crop_inds == None:
        print("Data is data clean 1")
        data_clean = data[chan_ind, :, :]

    else:
        print("Data is data clean 2")
        data_clean = data[
            chan_ind, crop_inds[0] : crop_inds[1], crop_inds[2] : crop_inds[3]
        ]

    if normalize:

        data_clean = (1 / 0.95) * data_clean * Maxs[ref_chan_ind] + Means[ref_chan_ind]

    print("Im here", data_clean.shape)
    return data_clean


class canvasHolder:
    def __init__(self, Zone, nb_lon, nb_lat, fpath="/home/brochetc/"):

        self.Zone = Zone
        self.X_min, self.Y_min = get_boundaries(Zone)
        self.nb_lon, self.nb_lat = nb_lon, nb_lat

        self.lonlat = extract_lonlat(fpath=fpath)

        self.Coords = [
            self.lonlat[0][
                self.Y_min : (self.Y_min + nb_lat), self.X_min : (self.X_min + nb_lon)
            ],
            self.lonlat[1][
                self.Y_min : (self.Y_min + nb_lat), self.X_min : (self.X_min + nb_lon)
            ],
        ]
        # print(self.Coords[0].shape, self.Coords[1].shape, self.lonlat[0].shape, self.lonlat[1].shape)

        self.proj0 = ccrs.Stereographic(central_latitude=46.7, central_longitude=2.0)

        self.proj_plot = ccrs.PlateCarree()
        self.axes_class = (GeoAxes, dict(map_projection=self.proj0))

    def project(self, padX=(5, 15), padY=(5, 5), ax=None):
        """

        create a plt.axes object using  a Stereographic projection
        Using grid indexes coordinates and width
        possibly reuses an existing ax object

        Inputs :

            X_min, Y_min : int, bottom left corner coordinates
            nb_lat, nb_lon : int, pixel width in latitude and longitude
            ax : None / plt.axes object, to be either manipulated or created

        Returns :

            ax : a plt.axes object ready to be filled with data,
            incorporating Borders and Coastline

        """

        if ax == None:
            ax = plt.axes(projection=self.proj0)

        Lat_min, Lon_min = grid_to_lat_lon(self.X_min - padX[0], self.Y_min - padY[0])
        Lat_max, Lon_max = grid_to_lat_lon(
            self.X_min + self.nb_lon + padX[1], self.Y_min + self.nb_lat + padY[1]
        )
        lon_bor = [Lon_min, Lon_max]
        lat_bor = [Lat_min, Lat_max]

        # Projecting boundaries onto stereographic grid
        lon_lat_1 = self.proj0.transform_point(
            lon_bor[0], lat_bor[0], ccrs.PlateCarree()
        )
        lon_lat_2 = self.proj0.transform_point(
            lon_bor[1], lat_bor[1], ccrs.PlateCarree()
        )

        # Properly redefining boundaries
        lon_bor = [lon_lat_1[0], lon_lat_2[0]]
        lat_bor = [lon_lat_1[1], lon_lat_2[1]]
        borders = lon_bor + lat_bor

        ax.set_extent(borders, self.proj0)  # map boundaries under the right projection

        return ax

    def makeDomainBox(
        self, output_dir, padX=(300, 30), padY=(180, 180), full_domain=False
    ):
        """
        plot the box of the studied subdomain on the AROME domain
        save figure on output dir
        """
        fig = plt.figure(figsize=(10, 10))
        grid0 = AxesGrid(
            fig,
            111,
            axes_class=self.axes_class,
            nrows_ncols=(1, 1),
            axes_pad=0.3,
            cbar_pad=0.25,
            cbar_location="right",
            cbar_mode="none",
            cbar_size="7%",
            label_mode="",
        )
        # grid0[0].add_patch(patches.Rectangle((self.X_min, self.Y_min), 128, 128, linewidth=10,edgecolor='r', facecolor='r'))

        ax = self.project(padX=padX, padY=padY, ax=grid0[0])
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"))  # adding coastline
        ax.add_feature(cfeature.BORDERS.with_scale("10m"))  # adding borders

        # properly drawing rectangle

        recCoords = grid_to_lat_lon(self.X_min, self.Y_min)
        recWidth = grid_to_lat_lon(self.X_min + self.nb_lon, self.Y_min)[1]
        recHeight = grid_to_lat_lon(self.X_min, self.Y_min + self.nb_lat)[0]

        lon_lat_rect = self.proj0.transform_point(
            recCoords[1], recCoords[0], ccrs.PlateCarree()
        )

        lon_lat_ext = self.proj0.transform_point(
            recWidth, recHeight, ccrs.PlateCarree()
        )

        ax.add_patch(
            patches.Rectangle(
                (lon_lat_rect[0], lon_lat_rect[1]),
                width=lon_lat_ext[0] - lon_lat_rect[0],
                height=lon_lat_ext[1] - lon_lat_rect[1],
                linewidth=5,
                edgecolor="r",
                facecolor="none",
                transform=self.proj0,
            )
        )

        # adding AROME full domain coords

        X_arome = -12.0
        nb_lon_AROME = 1121
        Y_arome = 37.5
        nb_lat_AROME = 717  # 685 ?? <-- previously written 685

        arome_coords = grid_to_lat_lon(0, 0)

        recWidth = grid_to_lat_lon(nb_lon_AROME, 0)[1]
        recHeight = grid_to_lat_lon(0, nb_lat_AROME)[0]

        print(recWidth, recHeight)

        lon_lat_rect = self.proj0.transform_point(
            -8.18, arome_coords[0], ccrs.PlateCarree()
        )

        lon_lat_ext = self.proj0.transform_point(
            recWidth, recHeight, ccrs.PlateCarree()
        )

        print(lon_lat_ext[0] - lon_lat_rect[0])
        print(lon_lat_ext[1] - lon_lat_rect[1])

        ax.add_patch(
            patches.Rectangle(
                (lon_lat_rect[0], lon_lat_rect[1]),
                width=lon_lat_ext[0] - lon_lat_rect[0],
                height=lon_lat_ext[1] - lon_lat_rect[1],
                edgecolor="none",
                facecolor="b",
                alpha=0.2,
                transform=self.proj0,
            )
        )

        plt.savefig(output_dir + self.Zone + "domain.png")

    def plot_data_normal(
        self,
        data,
        var_names,
        plot_dir,
        pic_name,
        contrast=False,
        cvalues=None,
        title="",
    ):
        """

        use self-defined axes structures and projections to plot numerical data
        and save the figure in dedicated directory

        Inputs :

            data: np.array -> data to be plotted shape Samples x Channels x Lat x Lon
                          with  Channels being the number of variables to be plotted

            plot_dir : str -> the directory to save the figure in

            pic_name : str -> the name of the picture to be saved


            contrast : bool (optional) -> check if boundary values for plot
                                         shoud be imposed (same value for all variables)

            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
             for each variable]

            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude


        Returns :


        Note :

            last docstring review by C .Brochet 15/04/2022

        """

        fig = plt.figure(figsize=(18, 6))
        axes = {}
        ims = {}

        if contrast:
            assert cvalues is not None
            Datamin = cvalues[0]
            Datamax = cvalues[1]

        grid = AxesGrid(
            fig,
            111,
            axes_class=self.axes_class,
            nrows_ncols=(len(var_names), data.shape[0]),
            axes_pad=0.05,
            cbar_pad=0.2,
            cbar_location="right",
            cbar_mode="edge",
            cbar_size="5%",
            label_mode="",
        )
        coef = -0.5
        for ind in range(data.shape[0]):
            # plotting each sample

            data_plot = data[ind, :, :, :]

            for i, var in enumerate(var_names):
                print(var)
                Var = var[0]
                unit = var[1]
                print(Var)
                if Var == "t2m":
                    cmap = "Greys"
                elif Var == "rr":
                    cmap = cmapRR
                    print("chosen cmap RR")
                else:
                    cmap = "Greys"

                print("projecting")
                axes[Var + str(ind)] = self.project(ax=grid[i * data.shape[0] + ind])

                if not contrast:
                    ims[Var + str(ind)] = axes[Var + str(ind)].pcolormesh(
                        self.Coords[0],
                        self.Coords[1],
                        data_plot[i, :, :],
                        cmap=cmap,
                        alpha=1,
                        transform=self.proj_plot,
                    )
                else:
                    print("Colormeshing")
                    ims[Var + str(ind)] = axes[Var + str(ind)].pcolormesh(
                        self.Coords[0],
                        self.Coords[1],
                        data_plot[i, :, :],
                        cmap=cmap,
                        alpha=1,
                        vmin=Datamin[i] / 2.0,
                        vmax=Datamax[i] / 2.0,
                        transform=self.proj_plot,
                    )
                    # axes[Var+str(ind)].set_title(str(coef), fontsize = 23)

                # varTitle=Var+' ('+unit+')'
                if ind == 0:
                    # Title+=' GAN'
                    axes[Var + str(ind)].title.set_text("AROME-EPS")  # , '+varTitle')
                else:
                    axes[Var + str(ind)].title.set_text("GAN")  # , '+varTitle')
                # elif i==0:
                # axes[Var+str(ind)].title.set_text('GAN')
                axes[Var + str(ind)].add_feature(
                    cfeature.COASTLINE.with_scale("10m")
                )  # adding coastline
                axes[Var + str(ind)].add_feature(
                    cfeature.BORDERS.with_scale("10m")
                )  # adding borders

                if ind == 0:  # or Var=='rr':
                    print("INDEXE", ind)
                    # grid.cbar_axes[i].colorbar(ims[Var+str(ind)], format='%.0e')
                    grid.cbar_axes[i].colorbar(ims[Var + str(ind)])
                    grid.cbar_axes[i].tick_params(labelsize="23")

            coef = coef + 0.25
        Title = title
        st = fig.suptitle(Title, fontsize="25")
        fig.subplots_adjust(bottom=0.005, top=0.96, left=0.05, right=0.95)
        # st.set_y(0.98)
        fig.canvas.draw()
        # fig.tight_layout()
        plt.savefig(plot_dir + pic_name, dpi=400)
        plt.close()

        """cbar=grid.cbar_axes[i+3*ind].colorbar(ims[var+str(ind)])
        yvalues=np.linspace(data[i,:,:].min(), data[i,:,:].max(),15)
        cbar.ax.set_yticks(yvalues)
    
        ylabels=['{:.1f}'.format(np.float32(xa)) for xa in yvalues]
        cbar.ax.set_yticklabels(ylabels, va='center',fontsize=8)"""

    def plot_data_wind(
        self,
        data,
        plot_dir,
        pic_name,
        contrast=False,
        cvalues=(-1.0, 1.0),
        withQuiver=True,
    ):
        """

        use self-defined axes structures and projections to plot wind magnitude and
        direction

        Inputs :

            data: np.array -> data to be plotted shape Samples x 2 x Lat x Lon
                          with 0 dimension being 'meridian wind' and
                          1 dimension being 'zonal wind'

            plot_dir : str -> the directory to save the figure in

            pic_name : str -> the name of the picture to be saved


            contrast : bool (optional) -> check if boundary values for plot
                                         shoud be imposed (same value for all variables)

            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
             for each variable]

            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude


        Returns :


        Note :

            last docstring review by C .Brochet 15/04/2022

        """

        fig = plt.figure(figsize=(23, 9))

        axes = {}
        ims = {}

        grid = AxesGrid(
            fig,
            111,
            axes_class=self.axes_class,
            nrows_ncols=(1, data.shape[0]),
            axes_pad=0.9,
            cbar_mode="each",
            cbar_pad=0.05,
            label_mode="",
        )

        u, v = data[:, 0, :, :], data[:, 1, :, :]
        quiSub = u.shape[1] // 32  # subsampling ratio for arrow drawings

        module, direction = cartesian2polar(u, v)

        axes = {}
        ims = {}

        for ind in range(data.shape[0]):
            Title = "Wind magnitude (m/s)"
            axes["wind_mag" + str(ind)] = self.project(ax=grid[ind])
            if not contrast:
                ims["wind_mag" + str(ind)] = axes["wind_mag" + str(ind)].pcolormesh(
                    self.Coords[0],
                    self.Coords[1],
                    module[ind, :, :],
                    cmap="plasma",
                    alpha=1,
                    transform=self.proj_plot,
                )
            else:
                ims["wind_mag" + str(ind)] = axes["wind_mag" + str(ind)].pcolormesh(
                    self.Coords[0],
                    self.Coords[1],
                    module,
                    cmap="plasma",
                    alpha=1,
                    vmin=cvalues[0],
                    vmax=cvalues[1],
                    transform=self.proj_plot,
                )
            axes["wind_mag" + str(ind)].add_feature(
                cfeature.COASTLINE.with_scale("10m")
            )  # adding coastline
            axes["wind_mag" + str(ind)].add_feature(
                cfeature.BORDERS.with_scale("10m")
            )  # adding borders

            print("colorbarising")
            grid.cbar_axes[ind].colorbar(ims["wind_mag" + str(ind)])
            grid.cbar_axes[ind].tick_params(labelsize="23")

            if withQuiver:
                axes["wind_dir" + str(ind)] = self.project(ax=grid[ind])
                ims["wind_dir" + str(ind)] = axes["wind_dir" + str(ind)].quiver(
                    x=np.array(self.Coords[0])[::quiSub, ::quiSub],
                    y=np.array(self.Coords[1])[::quiSub, ::quiSub],
                    u=0.25
                    * module[ind, ::quiSub, ::quiSub]
                    * direction[0][ind, ::quiSub, ::quiSub],
                    v=0.25
                    * module[ind, ::quiSub, ::quiSub]
                    * direction[1][ind, ::quiSub, ::quiSub],
                    scale=32.0,
                    color="white",
                    transform=self.proj_plot,
                )
                Title = "Wind magnitude (m/s) and direction"

            if ind == 0:
                # Title+=' GAN'
                axes["wind_mag" + str(ind)].title.set_text(r"AROME-EPS")

            else:
                axes["wind_mag" + str(ind)].title.set_text(r"GAN")
            axes["wind_mag" + str(ind)].title.set_fontweight("bold")
            axes["wind_mag" + str(ind)].title.set_size("25")

        st = fig.suptitle(Title, fontsize="25")
        st.set_y(0.96)
        fig.canvas.draw()
        fig.tight_layout()
        plt.savefig(plot_dir + pic_name)
        plt.close()

    def plot_abs_error(
        self,
        data,
        var_names,
        plot_dir,
        pic_name,
        col_titles,
        contrast=True,
        cmap_wind="viridis",
        cmap_t="Reds",
        suptitle="",
    ):
        """

        use self-defined axes structures and projections to plot numerical data
        and save the figure in dedicated directory

        Inputs :

            data: np.array -> data to be plotted shape Samples x Channels x Lat x Lon
                          with  Channels being the number of variables to be plotted

            plot_dir : str -> the directory to save the figure in

            pic_name : str -> the name of the picture to be saved


            contrast : bool (optional) -> check if boundary values for plot
                                         shoud be imposed (same value for all variables)

            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
             for each variable]

            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude


        Returns :


        Note :

            last docstring review by C .Brochet 15/04/2022

        """

        fig = plt.figure(
            figsize=(4 * data.shape[0], 3 * len(var_names)), facecolor="white"
        )
        axes = {}
        ims = {}
        grid = AxesGrid(
            fig,
            111,
            axes_class=self.axes_class,
            nrows_ncols=(len(var_names), data.shape[0]),
            axes_pad=0.1,
            cbar_pad=0.25,
            cbar_location="right",
            cbar_mode="edge",
            cbar_size="7%",
            label_mode="",
        )
        coef = -0.5
        for ind in range(data.shape[0]):
            # plotting each sample

            data_plot = data[ind, :, :, :]

            for i, var in enumerate(var_names):
                # print(var)
                Var = var[0]
                unit = var[1]
                # print(Var)
                if Var == "t2m":
                    cmap = cmap_t
                elif Var == "rr":
                    cmap = cmapRR
                elif Var == "t850":
                    cmap = cmap_t
                elif Var == "z500":
                    cmap = "Blues"
                elif Var == "tpw850":
                    cmap = cmap_t
                else:
                    cmap = cmap_wind
                axes[Var + str(ind)] = self.project(ax=grid[(i) * data.shape[0] + ind])

                if not contrast:
                    ims[Var + str(ind)] = axes[Var + str(ind)].pcolormesh(
                        self.Coords[0],
                        self.Coords[1],
                        data_plot[i, :, :],
                        cmap=cmap,
                        alpha=1,
                        transform=self.proj_plot,
                    )
                else:
                    # print("I am here")
                    ims[Var + str(ind)] = axes[Var + str(ind)].pcolormesh(
                        self.Coords[0],
                        self.Coords[1],
                        data_plot[i, :, :],
                        cmap=cmap,
                        alpha=1,
                        vmin=data.min(axis=(0, 2, 3))[i],
                        vmax=data.max(axis=(0, 2, 3))[i],
                        transform=self.proj_plot,
                    )

                axes[Var + str(ind)].add_feature(
                    cfeature.COASTLINE.with_scale("10m")
                )  # adding coastline
                axes[Var + str(ind)].add_feature(
                    cfeature.BORDERS.with_scale("10m")
                )  # adding borders

                if i == 0:
                    axes[Var + str(ind)].set_title(col_titles[ind], fontsize=28)  # 33
                if ind == 0 or Var == "rr":
                    # print('INDEXE',ind)
                    # grid.cbar_axes[i].colorbar(ims[Var+str(ind)], format='%.0e')
                    add_unit = " (" + unit + ")" if unit != "" else ""
                    grid.cbar_axes[i].colorbar(ims[Var + str(ind)]).set_label(
                        label=Var + add_unit, size=40
                    )  # 33
                    grid.cbar_axes[i].tick_params(labelsize=30)  # 32

            coef = coef + 0.25
        # for ind in range(data.shape[0]):
        #    axes['None'+str(ind)] = self.project(ax=grid[ind])
        #    axes['None'+str(ind)].axis('off')
        #    axes['None'+str(ind)].set_title(col_titles[ind])

        # Title='3-fields states'
        fig.subplots_adjust(bottom=0.005, top=0.95, left=0.05, right=0.95)
        st = fig.suptitle(suptitle, fontsize="30")  # 36
        st.set_y(0.98)
        # st.set_y(0.98)
        fig.canvas.draw()

        # fig.tight_layout()
        plt.savefig(plot_dir + pic_name, dpi=400, bbox_inches="tight")
        # plt.close()

        """cbar=grid.cbar_axes[i+3*ind].colorbar(ims[var+str(ind)])
        yvalues=np.linspace(data[i,:,:].min(), data[i,:,:].max(),15)
        cbar.ax.set_yticks(yvalues)
    
        ylabels=['{:.1f}'.format(np.float32(xa)) for xa in yvalues]
        cbar.ax.set_yticklabels(ylabels, va='center',fontsize=8)"""

    def plot_abs_error_sev_cbar(
        self,
        data,
        var_names,
        plot_dir,
        pic_name,
        col_titles,
        contrast=True,
        cmap_wind="viridis",
        cmap_t="Reds",
        suptitle="",
    ):
        """

        use self-defined axes structures and projections to plot numerical data
        and save the figure in dedicated directory

        Inputs :

            data: np.array -> data to be plotted shape Samples x Channels x Lat x Lon
                          with  Channels being the number of variables to be plotted

            plot_dir : str -> the directory to save the figure in

            pic_name : str -> the name of the picture to be saved


            contrast : bool (optional) -> check if boundary values for plot
                                         shoud be imposed (same value for all variables)

            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
             for each variable]

            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude


        Returns :


        Note :

            last docstring review by C .Brochet 15/04/2022

        """

        fig = plt.figure(
            figsize=(4 * data.shape[0], 3 * len(var_names)), facecolor="white"
        )
        axes = {}
        ims = {}

        grid = AxesGrid(
            fig,
            111,
            axes_class=self.axes_class,
            nrows_ncols=(len(var_names), data.shape[0]),
            axes_pad=(0.75, 0.25),
            cbar_pad=0.1,
            cbar_location="right",
            cbar_mode="each",
            cbar_size="7%",
            label_mode="",
        )
        coef = -0.5
        for ind in range(data.shape[0]):
            # plotting each sample

            data_plot = data[ind, :, :, :]

            for i, var in enumerate(var_names):
                # print(var)
                Var = var[0]
                unit = var[1]
                # print(Var)
                if Var == "t2m":
                    cmap = cmap_t
                elif Var == "rr":
                    cmap = cmapRR
                elif Var == "t850":
                    cmap = "coolwarm"
                elif Var == "z500":
                    cmap = "Blues"
                elif Var == "tpw850":
                    cmap = "plasma"
                else:
                    cmap = cmap_wind
                axes[Var + str(ind)] = self.project(ax=grid[i * data.shape[0] + ind])

                ims[Var + str(ind)] = axes[Var + str(ind)].pcolormesh(
                    self.Coords[0],
                    self.Coords[1],
                    data_plot[i, :, :],
                    cmap=cmap,
                    alpha=1,
                    transform=self.proj_plot,
                )

                axes[Var + str(ind)].add_feature(
                    cfeature.COASTLINE.with_scale("10m")
                )  # adding coastline
                axes[Var + str(ind)].add_feature(
                    cfeature.BORDERS.with_scale("10m")
                )  # adding borders

                if i == 0:
                    axes[Var + str(ind)].set_title(col_titles[ind], fontsize=15)  # 33
                # if ind==0 or Var=='rr':
                # print('INDEXE',ind)
                # grid.cbar_axes[i].colorbar(ims[Var+str(ind)], format='%.0e')

                cb = grid.cbar_axes[i * data.shape[0] + ind].colorbar(
                    ims[Var + str(ind)]
                )
                if ind == data.shape[0] - 1 or Var == "rr":
                    cb.set_label(label=Var + " (" + unit + ")", size=20)  # 33
                grid.cbar_axes[i * data.shape[0] + ind].tick_params(labelsize=12)  # 32

            coef = coef + 0.25
        # for ind in range(data.shape[0]):
        #    axes['None'+str(ind)] = self.project(ax=grid[ind])
        #    axes['None'+str(ind)].axis('off')
        #    axes['None'+str(ind)].set_title(col_titles[ind])

        # Title='3-fields states'
        fig.subplots_adjust(
            bottom=0.005,
            top=0.92 if len(var_names) > 1 else 0.75,
            left=0.05,
            right=0.95,
        )
        st = fig.suptitle(suptitle, fontsize="20")  # 36
        st.set_y(0.98)
        fig.canvas.draw()

        # fig.tight_layout()
        plt.savefig(plot_dir + pic_name, dpi=400, bbox_inches="tight")


"""
#Fonction ajouté pour l'exemple ici
def plot_RR_AE(RR_original,RR_AE,Date,Reseau,ech,MB,Zone,dim):
        lonlat=extract_lonlat()
        nb_lon=128
        nb_lat=128
        proj=ccrs.Stereographic(central_latitude=46.7,central_longitude=2)
        proj_plot=ccrs.PlateCarree()
        X_min,Y_min=zonage_bord(Zone)
        axes_class= (GeoAxes,dict(map_projection=proj))
        fig=plt.figure(figsize=(10,4))
        grid= AxesGrid(fig, 111, axes_class=axes_class,nrows_ncols=(1,2),
                               axes_pad=0.3,cbar_location= 'right', cbar_mode= 'single',
                               cbar_pad=0.2,label_mode='')

        
        ax_RR=proj_multi(grid[0],X_min,Y_min,nb_lat,nb_lon)
        ax_AE=proj_multi(grid[1],X_min,Y_min,nb_lat,nb_lon)

        bounds = np.array([0, 0.1, 0.4,0.6,1.2,2.1,3.6,6.5,12,21,36,65,120,205,360])
        legende= ["0","0.1","0.4","0.6","1.2","2.1","3.6","6.5","12","21","36","65","120","205","360"]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=14)
        cmap2 = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow",
                                                     "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None)    

        ax_RR.pcolormesh(lonlat[0][Y_min:(Y_min+nb_lat),X_min:(X_min+nb_lon)],lonlat[1][Y_min:(Y_min+nb_lat),X_min:(X_min+nb_lon)],RR_original,cmap=cmap2,norm=norm,alpha=1,transform=proj_plot)
        im=ax_AE.pcolormesh(lonlat[0][Y_min:(Y_min+nb_lat),X_min:(X_min+nb_lon)],lonlat[1][Y_min:(Y_min+nb_lat),X_min:(X_min+nb_lon)],RR_AE,cmap=cmap2,norm=norm,alpha=1,transform=proj_plot)
        cbar=grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_yticks(np.linspace(0, 361, 15))
        cbar.ax.set_yticklabels(legende, va='center',fontsize=8)

        date_format=day_and_hour_UTC(Date,Reseau,ech)
        fig.suptitle("Plot RR1h et AE (dim "+str(dim)+"), Date: "+str(Date)+", Run: "+str(Reseau)+"h, MB : "+MB+" \n Zone : "+Zone+", Validite : "+date_format+" UTC",fontsize=10)
        path_plot='/home/mouniera/Documents/Scenario_Megabase/Plot_AE/Plot_test/'+Date+Reseau+'/dim'+str(dim)+'/Zone_'+Zone+'/'+MB+'/'
        test_and_create_path(path_plot)
        plt.savefig(path_plot+'Test_'+date_format.replace(" ","_").replace("/","-")+'_dim'+str(dim)+'.png')
        plt.close()
        
    #bounds = np.linspace(-1.,1.,15)
    #norm = colors.BoundaryNorm(boundaries=bounds, ncolors=14)
    #cmap2 = colors.ListedColormap(["white","mediumpurple","blue","dodgerblue","darkseagreen","seagreen","greenyellow","yellow",
    #                                             "navajowhite","sandybrown","darkorange","red","darkred","black"], name='from_list', N=None) 
    
    
    
    
    #axes['wind']=proj_multi(grid[0], X_min,Y_min,nb_lat,nb_lon)
    #ims['wind']=axes['wind'].quiver(x=Coords[0][::4],y=Coords[1][::4],u=data[0,::4,::4],v=data[1,::4,::4],scale=4)    """
