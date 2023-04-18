import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.sparse as sp


def cart(theta, phi):
    return np.vstack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))).transpose()

def sub_lattice(N, ulat, ulon, llat, llon):

    N = int(N)
    ulat = (ulat + 90) * np.pi / 180
    llat = (llat + 90) * np.pi / 180
    ulon = (ulon + 180) * np.pi / 180
    llon = (llon + 180) * np.pi / 180

    surface = np.abs((ulon - llon) *(np.cos(llat) - np.cos(llon)))
    new_N = np.pi * (2.728 ** 2) * N / surface
    new_N = int(new_N) + 1

    nodes = sphericallattice_Fibonacci(new_N, cartesian=False)
    index = (nodes[:, 0] <= ulon) & (llon <= nodes[:,0]) & (nodes[:, 1] <= ulat) & (llat <= nodes[:,1])

    new_N *= (N / sum(index))
    new_N = int(new_N) + 1

    nodes = sphericallattice_Fibonacci(new_N, cartesian=False)
    index = (nodes[:, 0] <= ulon) & (llon <= nodes[:, 0]) & (nodes[:, 1] <= ulat) & (llat <= nodes[:, 1])

    return sphericallattice_Fibonacci(new_N, cartesian=False)[index]


def sphericallattice_Fibonacci(N, cartesian = True):

    theta = 2 * np.pi * (1 - 2 / (1 + np.sqrt(5))) * np.arange(1, N+1, 1)
    phi = np.arccos(1 - 2 * np.arange(1, N+1, 1) / N)

    if cartesian:
        return cart(theta, phi)
    else :
        return np.vstack((theta % (2 * np.pi), phi % (2 * np.pi))).transpose()

def testreal(x):
    return isinstance(x, float) or isinstance(x, int)

def cartmesh(theta, phi):

    return np.array([np.outer(np.cos(theta), np.sin(phi)), np.outer(np.sin(theta), np.sin(phi)),
            np.outer(np.ones(np.size(theta)), np.cos(phi))])

def splot(xyz, c, title="surprise"):

    if type(c) is not list:
        c = [c]
    fig, axes = plt.subplots(ncols=len(c), nrows=1, subplot_kw=dict(projection='3d'), figsize=(25, 12))

    if len(c) > 1:
        for i in range(len(c)):
            axes[i].plot_surface(xyz[0], xyz[1], xyz[2], linewidth=0.0, cstride=1, rstride=1, facecolors=cm.jet(c[i]))
            axes[i].set_title(title[i])
    else:
        axes.plot_surface(xyz[0], xyz[1], xyz[2], linewidth=0.0, cstride=1, rstride=1, facecolors=cm.jet(c[0]))
        axes.set_title(title)

def splot_scatter(long, lat, c, title="surprise", llon=-180, llat=-70, ulon=180, ulat=70, return_=False, frame=None,
                  focus=False):

    jet = plt.get_cmap('jet', 2000)

    if type(c) is not list:
        c = [c]
    fig, axes = plt.subplots(ncols=len(c), nrows=1, figsize=(25, 12))

    if len(c) > 1:
        for i in range(len(c)):
            if focus:
                if type(frame) == np.ndarray:
                    map = Basemap(projection='cyl', llcrnrlon=frame[0, 0], llcrnrlat=frame[1, 0],
                                  urcrnrlon=frame[0, 1], urcrnrlat=frame[1, 1], ax=axes[i])
            else:
                map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                              urcrnrlon=ulon, urcrnrlat=ulat, ax=axes[i])
            x, y = map(long, lat)
            map.drawcoastlines()
            cnorm = colors.Normalize(vmin=np.min(c[i]), vmax=np.max(c[i]))
            scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=jet)
            map.scatter(x, y, color=scalarMap.to_rgba(c[i].reshape(-1)))

            if type(frame) == np.ndarray:
                x_boundary, y_boundary = map(
                    np.array([frame[0, 0], frame[0, 1], frame[0, 1], frame[0, 0], frame[0, 0]]),
                    np.array([frame[1, 1], frame[1, 1], frame[1, 0], frame[1, 0], frame[1, 1]]))
                map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

            plt.colorbar(mappable=scalarMap, ax=axes[i])
            axes[i].set_title(title[i])
    else:
        if focus:
            if type(frame) == np.ndarray:
                map = Basemap(projection='cyl', llcrnrlon=frame[0, 0], llcrnrlat=frame[1, 0],
                              urcrnrlon=frame[0, 1], urcrnrlat=frame[1, 1], ax=axes)
        else:
            map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                          urcrnrlon=ulon, urcrnrlat=ulat, ax=axes)
        x, y = map(long, lat)
        map.drawcoastlines()
        cnorm = colors.Normalize(vmin=np.min(c[0]), vmax=np.max(c[0]))
        scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=jet)
        map.scatter(x, y, color=scalarMap.to_rgba(c[0].reshape(-1)))

        if type(frame) == np.ndarray:
            x_boundary, y_boundary = map(np.array([frame[0, 0], frame[0, 1], frame[0, 1], frame[0, 0], frame[0, 0]]),
                                         np.array([frame[1, 1], frame[1, 1], frame[1, 0], frame[1, 0], frame[1, 1]]))
            map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

        plt.colorbar(mappable=scalarMap, ax=axes)
        axes.set_title(title)

    plt.show()
    if return_:
        return map


def splot_grid(latlong, c, title="surprise", llon=-180, llat=-70, ulon=180, ulat=70, type_plot="pcolormesh",
               focus=False, frame=None, vmin=None, vmax=None, cmap=None, display=True, mask_land=False):

    plt.ioff()
    if type(c) is not list:
        c = [c]
    fig, axes = plt.subplots(ncols=len(c), nrows=1, figsize=(25, 12))

    if len(c) > 1:
        for i in range(len(c)):
            if focus:
                if type(frame) == np.ndarray:
                    map = Basemap(projection='cyl', llcrnrlon=frame[0, 0], llcrnrlat=frame[1, 0],
                                  urcrnrlon=frame[0, 1], urcrnrlat=frame[1, 1], ax=axes[i])
            else:
                map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                              urcrnrlon=ulon, urcrnrlat=ulat, ax=axes[i])
            x, y = map(latlong[1], latlong[0])
            map.drawcoastlines()
            if type_plot == "pcolormesh":
                if isinstance(cmap, colors.LinearSegmentedColormap) and testreal(vmin) and testreal(vmax):
                    im = map.pcolormesh(x, y, c[i], shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
                else:
                    im = map.pcolormesh(x, y, c[i], shading="auto")
                plt.colorbar(im, ax=axes[i], shrink=0.3)
                if type(frame) == np.ndarray:
                    x_boundary, y_boundary = map(
                        np.array([frame[0, 0], frame[0, 1], frame[0, 1], frame[0, 0], frame[0, 0]]),
                        np.array([frame[1, 1], frame[1, 1], frame[1, 0], frame[1, 0], frame[1, 1]]))
                    map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)
            if type_plot=="contourf":
                im = map.contourf(x, y, c[i], 50)
                plt.colorbar(im, ax=axes[i], shrink=0.3)
                if type(frame) == np.ndarray:
                    x_boundary, y_boundary = map(
                        np.array([frame[0, 0], frame[0, 1], frame[0, 1], frame[0, 0], frame[0, 0]]),
                        np.array([frame[1, 1], frame[1, 1], frame[1, 0], frame[1, 0], frame[1, 1]]))
                    map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

            axes[i].set_title(title[i])
            if mask_land:
                map.fillcontinents(color="k")
    else:
        if focus:
            if type(frame) == np.ndarray:
                map = Basemap(projection='cyl', llcrnrlon=frame[0, 0], llcrnrlat=frame[1, 0],
                              urcrnrlon=frame[0, 1], urcrnrlat=frame[1, 1], ax=axes)
        else:
            map = Basemap(projection='cyl', llcrnrlon=llon, llcrnrlat=llat,
                          urcrnrlon=ulon, urcrnrlat=ulat, ax=axes)
        x, y = map(latlong[1], latlong[0])
        map.drawcoastlines()
        if type_plot == "pcolormesh":
            if isinstance(cmap, colors.LinearSegmentedColormap) and testreal(vmin) and testreal(vmax):
                im = map.pcolormesh(x, y, c[0], shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                im = map.pcolormesh(x, y, c[0], shading="auto")
            plt.colorbar(im, ax=axes)
            if type(frame) == np.ndarray:
                x_boundary, y_boundary = map(
                    np.array([frame[0, 0], frame[0, 1], frame[0, 1], frame[0, 0], frame[0, 0]]),
                    np.array([frame[1, 1], frame[1, 1], frame[1, 0], frame[1, 0], frame[1, 1]]))
                map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)
        if type_plot == "contourf":
            im = map.contourf(x, y, c[0], 50)
            plt.colorbar(im, ax=axes)
        axes.set_title(title)
        if mask_land:
            map.fillcontinents(color="k")
        if type(frame) == np.ndarray:
            x_boundary, y_boundary = map(np.array([frame[0, 0], frame[0, 1], frame[0, 1], frame[0, 0], frame[0, 0]]),
                                         np.array([frame[1, 1], frame[1, 1], frame[1, 0], frame[1, 0], frame[1, 1]]))
            map.plot(x_boundary, y_boundary, color='m', linewidth=10, marker=None)

    plt.ion()
    if display:
        plt.show()
    if not display:
        return [fig, axes]


def saveplot(xyz, c, title="surprise", filename=None):

    if type(c) is not list:
        c = [c]
    fig, axes = plt.subplots(ncols=len(c), nrows=1, subplot_kw=dict(projection='3d'), figsize=(25,12))
    plt.ion()

    if len(c) > 1:
        for i in range(len(c)):
            axes[i].plot_surface(xyz[0], xyz[1], xyz[2], linewidth=0.0, cstride=1, rstride=1, facecolors=cm.jet(c[i]))
            axes[i].set_title(title[i])
    else:
        axes.plot_surface(xyz[0], xyz[1], xyz[2], linewidth=0.0, cstride=1, rstride=1, facecolors=cm.jet(c[0]))
        axes.set_title(title)

    plt.ioff()
    plt.savefig(filename)
    plt.close(fig)

def min_increment (x):

    x2 = x[1::]
    x1 = x[0:-1]
    x3 = x2-x1

    return min(x3)

def proj_circle(T, x):
    return np.vstack((np.cos((2 * np.pi / T) * x), np.sin((2 * np.pi / T) * x))).transpose()

#def video(image_folder = 'frames1', video_name = 'video.avi'):

 #   images = []
  #  for i in range(400):
   #     images.append("frame" + str(i) + ".png")
    #frame = cv2.imread(os.path.join(image_folder, images[0]))
    #height, width, layers = frame.shape

    #video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    #for image in images:
     #   video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    #video.release()

def vectortomatrix (vector, image_dim):

    matrice = sp.lil_matrix(sp.diags(np.zeros(vector.shape[0])))
    matrice = matrice[0:image_dim, :]

    i=0
    for j in range(vector.shape[0]):
        if vector[j]:
            matrice[i, j] = 1
            i += 1

    return matrice
