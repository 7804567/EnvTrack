import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import os
import tqdm
from utility import *
from utility_fun import *
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
plt.rcParams.update({'font.size':20})

class sfd_graph():

    def __init__(self, model):
        self.model = model

    def fetch_sev(self, xyz, t, code, data, normalize_mean=None):

        eval = np.zeros(int(xyz.shape[0] * xyz.shape[1] / 3))
        new_code = ""

        if type(normalize_mean) == str and (normalize_mean == "seas" or normalize_mean == "trend"):
            mean = self.model.mean(xyz, code=normalize_mean)

        if "trend" in code:
            eval += self.model.evaluation(xyz, t, code=["trend"]).reshape(-1)
            if normalize_mean == "seas":
                eval += mean
            elif normalize_mean == "trend":
                eval -= mean
            new_code += "trend+"

        if "seas" in code:
            eval += self.model.evaluation(xyz, t, code=["seas"]).reshape(-1)
            if normalize_mean == "seas":
                eval -= mean
            elif normalize_mean == "trend":
                eval += mean
            new_code += "seas+"

        if "trendkernel" in code:
            eval += self.model.evaluation(xyz, t, code=["trendkernel"]).reshape(-1)
            new_code += "trendkernel+"

        if "trendkernel" in self.model.dims:
            for dim in range(self.model.dims["trendkernel"]["time"]):
                if "trendkernel" + str(dim+1) in code:
                    eval += self.model.evaluation(xyz, t, code=["trendkernel" + str(dim+1)]).reshape(-1)
                    new_code += "trendkernel" + str(dim+1) + "+"

        if "data" in code:
            eval = data
            new_code = "data"

        if "residuals" in code:
            full_code = []
            if hasattr(self.model, "weights_seas"):
                full_code.append("seas")
            if hasattr(self.model, "weights_trend"):
                full_code.append("trend")
            if hasattr(self.model, "weights_trendkernel"):
                full_code.append("trendkernel")
            eval = data
            eval -= self.model.evaluation(xyz, t, code=full_code).reshape(-1)
            new_code = "residuals"

        title = "Spatial " + new_code + " field estimate at time " + str(round(t, 3)) + "."

        return eval, new_code, title

    def spatial_plot(self, t, code,
                     normalize_mean=None, display=True, type_plot="pcolormesh",
                     llat=-60, llon=-180, ulat=70, ulon=180, vmin=None, vmax=None,
                     cmap=None, N=80, mask_land=False):


        data=None
        latlong = np.meshgrid(np.linspace(0, np.pi, N), np.linspace(0, 2 * np.pi, 2 * N))
        lat2, long2 = latlong[0].reshape(-1), latlong[1].reshape(-1)
        xyz = cart(long2, lat2)
        latlong = np.array([(latlong[0] - np.pi / 2) * 180 / np.pi, (latlong[1] - np.pi) * 180 / np.pi])

        reshape_loc = lambda e:  e.reshape(2 * N, N)

        c, title = [], []
        for i in range(len(code)):
            eval, new_code, title_ = self.fetch_sev(xyz, t, code[i], data, normalize_mean)
            eval = reshape_loc(eval)
            c.append(eval)
            title.append(title_)

        plot = splot_grid(latlong, c, title, llon=llon, llat=llat, ulon=ulon, ulat=ulat, focus=False,
                                type_plot=type_plot, frame=None, vmin=vmin, vmax=vmax, cmap=cmap,
                                display=display, mask_land=mask_land)

        if display:
            return 0
        else:
            return plot

    def fetch_ev(self, xyz, sample, code, linewidth=None, normalize_mean=None):

        eval = np.zeros(sample.shape[0])
        new_code = ""

        if normalize_mean == "seas" or normalize_mean == "trend":
            mean = self.model.mean(xyz, code=normalize_mean)

        if "trend" in code:
            eval += self.model.evaluation(xyz, sample, code=["trend"])
            if normalize_mean == "seas":
                eval += mean
            elif normalize_mean == "trend":
                eval -= mean
            new_code = new_code + "trend+"

        if "seas" in code:
            eval += self.model.evaluation(xyz, sample, code=["seas"])
            if normalize_mean == "seas":
                eval -= mean
            elif normalize_mean == "trend":
                eval += mean
            new_code = new_code + "seas+"

        if "trendkernel" in code:
            eval += self.model.evaluation(xyz, sample, code=["trendkernel"])
            new_code = new_code + "trendkernel+"

        if linewidth is None:
            linewidth = 1

        return sample, eval, linewidth, new_code

    def temporal_plot(self, longlat, time, code=[],
                      normalize_mean="none", geo_pos=True, linewidth=[]):

        xyz = cart(longlat[0], longlat[1]).reshape(-1)
        sample = np.linspace(time[0], time[1], 1000)

        if len(linewidth) < len(code):
            linewidth = [1] * len(code)

        if geo_pos:
            fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(25, 12),
                                     gridspec_kw={
                           'width_ratios': [8, 16]})
            ax_plot = axes[1]
            map = Basemap(projection='ortho', lat_0=longlat[1], lon_0=longlat[0], ax=axes[0])
            map.drawmapboundary(fill_color='aqua')
            map.fillcontinents(color='coral', lake_color='aqua')
            map.drawcoastlines()
            xx, yy = map(longlat[0], longlat[1])
            map.plot(xx, yy, marker="D", color="k")
            axes[0].set_title("Position (LON:"
                                + str(round(longlat[0], 2)) + ", LAT:" + str(round(longlat[1], 2)) + ") on earth.")

        else:
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 12))
            ax_plot = axes

        for i in range(len(code)):
            s, eval, lw, new_code = self.fetch_ev(xyz, sample, code[i], linewidth=linewidth[i], normalize_mean=normalize_mean)
            if "residuals+" in code or "data+" in code:
                ax_plot.scatter(s, eval, linewidth=lw, label=new_code)
            else:
                ax_plot.plot(s, eval, linewidth=lw, label=new_code)

        ax_plot.set_xlabel("Time")
        ax_plot.set_ylabel("Data unit")
        ax_plot.set_title("Graph of different reconstructions of the signal at (LON:"
                              + str(round(longlat[0], 2)) + ", LAT:" + str(round(longlat[1], 2)) + ").")
        ax_plot.legend()

    def spatialRes_plot(self, indice, llon=-180, llat=-70, ulon=180, ulat=70):

        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 12))
        map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=ulon, urcrnrlat=ulat, ax=axes)
        map.drawcoastlines()
        if len(indice) == 2:
            day_start = indice[0]
            day_end = indice[1]
        elif len(indice) == 1:
            day_start = indice[0]
            day_end = indice[0]
        else:
            print("Error; indice must have only one or two indexes")

        index = (self.model.data.time >= day_start) & (self.model.data.time <= day_end)
        long_plot = self.model.data.long[index]
        lat_plot = self.model.data.lat[index]
        temp_plot = self.model.data.temp[index]
        fitted = self.model.F(self.model.estimate)[index]
        temp_plot -= fitted

        x, y = map(long_plot, lat_plot)
        jet = plt.get_cmap('jet', 2000)
        cnorm = colors.Normalize(vmin=np.min(temp_plot), vmax=np.max(temp_plot))
        scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=jet)
        im = map.scatter(x, y, marker='D', color=scalarMap.to_rgba(temp_plot))
        plt.colorbar(mappable=scalarMap, ax=axes)
        axes.set_title("ARGO residuals between day " + str(day_start) + " and " + str(day_end) + " (included) at pressure " \
                       + str(self.model.data.pressure) + ".")
        map.fillcontinents(color="k")
        plt.show()


