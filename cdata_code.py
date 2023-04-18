import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
from utility import *

plt.rcParams.update({'font.size':20})

class cdata:

    def __init__(self, year_start, year_end, pressure):

        self.pressure = pressure
        domain = os.path.join(os.path.join(os.getcwd(), "ARGO"), "compact"+str(pressure))
        if os.path.exists(domain):
            time, lat, long, temp, pres = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            time_list, lat_list, long_list, temp_list, pres_list, time_dic = [], [], [], [], [], {}

            for year in range(year_start, year_end + 1):
                loc_time, loc_lat, loc_long, loc_temp, loc_pres = \
                    np.load(os.path.join(domain, str(year)+".npy"), allow_pickle=True)
                time = np.hstack((time, loc_time))
                lat = np.hstack((lat, loc_lat))
                long = np.hstack((long, loc_long))
                temp = np.hstack((temp, loc_temp))
                pres = np.hstack((pres, loc_pres))
            current_indice = 0
            while current_indice < time.shape[0]:
                old_day = int(time[current_indice])
                new_indice = current_indice
                current_day = int(time[current_indice])
                while current_day == old_day and new_indice < time.shape[0]:
                    new_indice += 1
                    if new_indice > time.shape[0]:
                        current_day, old_day = int(time[new_indice-1]), current_day
                    else:
                        current_day, old_day = int(time[new_indice-1]), current_day
                #time_dic[old_day] = len(time_dic)
                time_list.append(time[current_indice:new_indice])
                lat_list.append(lat[current_indice:new_indice])
                long_list.append(long[current_indice:new_indice])
                temp_list.append(temp[current_indice:new_indice])
                pres_list.append(pres[current_indice:new_indice])
                current_indice = new_indice

            time_anchor = np.min(time)
            for i in range(len(time_list)):
                time_list[i] -= time_anchor

            self.time_anchor = time_anchor
            self.time = time
            self.lat = lat
            self.long = long
            self.temp = temp
            self.pres = pres
            self.xyz = cart((self.long + 180) * np.pi / 180, (self.lat + 90) * np.pi / 180)
            self.true_dim = self.temp.shape[0]
            self.pressure = pressure
            self.year_start = year_start
            self.year_end = year_end

            self.time_list = time_list
            self.lat_list = lat_list
            self.long_list = long_list
            self.temp_list = temp_list
            self.pres_list = pres_list
            #self.time_dic = time_dic

        else:
            print("The data has not been compacted.")


    def spatial_plot(self, indice, llon=-180, llat=-70, ulon=180, ulat=70):

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
        temp_plot = np.array([])
        lat_plot = np.array([])
        long_plot = np.array([])
        for day in range(day_start, day_end + 1):
            try:
                temp_plot = np.hstack((temp_plot, self.temp_list[day]))
                lat_plot = np.hstack((lat_plot, self.lat_list[day]))
                long_plot = np.hstack((long_plot, self.long_list[day]))
            except:
                donothing = 1

        x, y = map(long_plot,  lat_plot)
        jet = plt.get_cmap('jet', 2000)
        cnorm = colors.Normalize(vmin=np.min(temp_plot), vmax=np.max(temp_plot))
        scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=jet)
        im = map.scatter(x, y, marker='D', color=scalarMap.to_rgba(temp_plot))
        plt.colorbar(mappable=scalarMap, ax=axes)
        axes.set_title("ARGO data between day " + str(day_start) + " and " + str(day_end) + " (included) at pressure "\
                       + str(self.pressure) + ".")
        map.fillcontinents(color="k")
        plt.show()


