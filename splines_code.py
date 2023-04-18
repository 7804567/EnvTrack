from pycsou.math.green import Matern
from utility import *

class spline:

    def __init__(self, specification="matern1.5", eps=1):

        self.specification = specification
        self.eps = eps
        self.kernel = False

        if specification == "matern0.5":
            self.evaluate = Matern(0, self.eps)
            self.primitive = lambda r: -self.eps * np.exp(-r / self.eps)
        elif specification == "matern1.5":
            self.evaluate = Matern(1, self.eps)
            self.primitive = lambda r: ((-2 * self.eps / np.sqrt(3)) - r) * np.exp(- r * np.sqrt(3) / self.eps)
        elif specification == "matern2.5":
            self.evaluate = Matern(2, self.eps)
        elif specification == "matern3.5":
            self.evaluate = Matern(3, self.eps)
        elif specification == "D2":
            self.evaluate = lambda x: np.maximum(0, x)
            self.primitive = lambda x: (np.maximum(0, x) ** 2) / 2
            self.kernel = True
            self.dim_kernel = 2
            self.evaluate_kernel = lambda x: np.vstack((x * 0 + 1, x))
            self.integrate_kernel = lambda x: np.vstack((np.hstack((x[1:] - x[:-1], x[-1] - x[-2])),
                                                         np.hstack(((x[1:] ** 2 - x[:-1] ** 2) / 2,
                                                                    ((2 * x[-1] - x[-2]) ** 2 - x[-1] ** 2) / 2))
                                                         ))
        elif specification == "triangle":
            #the support is of size 2 eps, -eps to 0 to eps.
            self.evaluate = lambda x: np.maximum(0, 1 - (np.abs(x) / self.eps))
            self.primitive = lambda x: ((x >= -self.eps) & (x <= self.eps)) * \
                                       (x - np.sign(x) * ((x ** 2) / (2 * self.eps)) + (self.eps / 2)) + \
                                self.eps * (x > self.eps)
            self.kernel = True
            self.dim_kernel = 2
            self.evaluate_kernel = lambda x: np.vstack((x * 0 + 1, x))
            self.integrate_kernel = lambda x: np.vstack((np.hstack((x[1:] - x[:-1], x[-1] - x[-2])),
                                                         np.hstack(((x[1:] ** 2 - x[:-1] ** 2) / 2,
                                                                    ((2 * x[-1] - x[-2]) ** 2 - x[-1] ** 2) / 2))
                                                         ))
        else:
            self.evaluate = "function not defined"

    def new_eps(self, eps):
        self.eps = eps
        if self.specification == "matern0.5":
            self.evaluate = Matern(0,self.eps)
        elif self.specification == "matern1.5":
            self.evaluate = Matern(1,self.eps)
        elif self.specification == "matern2.5":
            self.evaluate = Matern(2,self.eps)
        elif self.specification == "matern3.5":
            self.evaluate = Matern(3,self.eps)
        else:
            self.evaluate = "function not defined"

class spline_field(spline):

    def __init__(self, spline, nodes=None, domain="spherical", size=10, frame=None):

        #domain can be "spherical", "circular" or a positive float
        #if domain is a positive float, it is assumed that this number corresponds to the size of
            #the interval considered in the form [0, domain]
        self.domain = domain

        if spline.specification == "D2" and type(domain) != np.ndarray:
            raise ValueError("D2 can only be used for real splines.")


        if type(domain) != np.ndarray:
            if self.domain == "spherical":
                # variable framed is here to take into account a possible restriction of the sphere to a section of it
                # framed should be in defined, by the user, as np.array([[llon, ulon],[llat, ulat]])
                self.frame = frame

        if type(nodes) != np.ndarray and type(domain) != np.ndarray:

            if self.domain == "spherical":
                self.size = size
                self.nodes_SC = sphericallattice_Fibonacci(self.size, cartesian=False)
                self.nodes = cart(self.nodes_SC[:, 0], self.nodes_SC[:, 1])

            if self.domain == "circular":
                self.size = size
                self.nodes_angle = np.linspace(0, 2 * np.pi, self.size + 1)[0:self.size]
                self.nodes = np.vstack((np.cos(self.nodes_angle), np.sin(self.nodes_angle))).transpose()

        elif type(nodes) != np.ndarray and isinstance(self.domain, np.ndarray):
            self.size = size
            self.nodes = np.linspace(self.domain[0], self.domain[1], size)

        else:
            if type(self.domain) == np.ndarray:
                self.nodes = nodes
            else:
                if self.domain == "circular":
                    self.nodes = nodes
                else:
                    self.nodes_SC = nodes
                    self.nodes = cart(self.nodes_SC[:, 0], self.nodes_SC[:, 1])
            self.size = self.nodes.shape[0]

        #x has to be a real number or a matrix of real numbers ( not a vector )
        if isinstance(self.domain, np.ndarray):
            if "matern" in spline.specification:
                self.proj = lambda x: np.abs(x)
            elif spline.specification == "D2":
                self.proj = lambda x: x
            elif spline.specification == "triangle":
                self.proj = lambda x: x
        else:
            if domain == "spherical":
                self.proj = lambda x: np.sqrt(2 - (2 * x))

            if domain == "circular":
                if "matern" in spline.specification:
                    self.proj = lambda x: np.sqrt(2 - (2 * x))
                elif spline.specification == "triangle":
                    self.proj = lambda x: np.arccos(np.clip(x, -1, 1))

        super(spline_field, self).__init__(specification=spline.specification, eps=spline.eps)

    def ev_field(self, pts, coeff):

        if isinstance(self.domain, np.ndarray):
            if testreal(pts):
                pts = np.array(pts)
            return self.evaluate(self.proj(pts.reshape(pts.shape[0], 1) - self.nodes.reshape(1, self.size))).dot(coeff)

        else:
            if self.size <= 1000:
                return self.evaluate(self.proj(pts.dot(self.nodes.transpose()))).dot(coeff)

            else:

                bachs_number = int(self.size / 1000)
                evaluation = np.zeros(pts.shape[0])
                for i in range(bachs_number):
                    evaluation += \
                        self.evaluate(self.proj(pts.dot(self.nodes[1000 * i: 1000 * (i + 1)].transpose()))).\
                        dot(coeff[1000 * i: 1000 * (i + 1)])
                    print("Evaluation finished at " + str(round((i + 1) * 1000 / self.size, 3) * 100) + "%.")
                evaluation += \
                    self.evaluate(self.proj(pts.dot(self.nodes[1000 * bachs_number:].transpose()))).\
                        dot(coeff[1000 * bachs_number:])
                return evaluation

    def int_on_line(self, pts, coeff="auto"):

        if not isinstance(self.domain, np.ndarray):
            raise Exception("int_on_line can only be used to integrate real splines.")
        else:
            if coeff == "auto":
                coeff = np.ones(self.size)
        pts = np.hstack((pts, pts[-1] + (pts[-1] - pts[-2])))
        start = pts[:-1].reshape((pts.shape[0] - 1, 1)) - self.nodes.reshape((1, self.size))
        end = pts[1:].reshape((pts.shape[0] - 1, 1)) - self.nodes.reshape((1, self.size))
        if "matern" in self.specification:
            return ((end >= 0) * self.primitive(end) - (end >= 0) * (start >= 0) * self.primitive(start) + \
                   (end >= 0) * (start < 0) * (self.primitive(-start) - 2 * self.primitive(0)) + \
                   (end < 0) * (start < 0) * (self.primitive(-start) - self.primitive(-end))) * coeff
        if "D2" in self.specification:
            return self.primitive(end) - self.primitive(start)
        if "triangle" in self.specification:
            return self.primitive(end) - self.primitive(start)

    def homogeneity(self, type_plot="scatter", llat=-90, ulat=90, llon=-180, ulon=180, plot_pos=False, focus=False):

        if isinstance(self.domain, np.ndarray):
            test = np.linspace(self.domain[0], self.domain[1], 1000)
            plt.plot(test, self.ev_field(test, np.ones(self.size)))

        else:

            if self.domain == "circular":
                test = np.linspace(0, 2 * np.pi, 1000)
                pts = np.vstack((np.cos(test), np.sin(test))).transpose()
                ev_pts = self.ev_field(pts, np.ones(self.size))
                plt.plot(pts[:, 0] * (1 + ev_pts), pts[:, 1] * (1 + ev_pts))
                plt.plot(pts[:, 0], pts[:, 1])

            if self.domain == "spherical":

                if type_plot == "pcolormesh" or type_plot == "contourf" or type_plot == "scatter":

                    lat = np.linspace(0, np.pi, 100)
                    long = np.linspace(0, 2 * np.pi, 200)
                    latlong = np.meshgrid(lat, long)
                    lat2, long2 = latlong[0].reshape(-1), latlong[1].reshape(-1)
                    xyz = cart(long2, lat2)

                    if type_plot == "pcolormesh":
                        c = self.ev_field(xyz, np.ones(self.size)).reshape(200, 100)
                        map = splot_grid(np.meshgrid((lat - np.pi / 2) * 180 / np.pi, (long - np.pi) * 180 / np.pi),
                                c, "Homogeneity", llat=llat, ulat=ulat, llon=llon, ulon=ulon, type_plot="pcolormesh",
                                         frame=self.frame, focus=focus)

                    if type_plot == "contourf":
                        c = self.ev_field(xyz, np.ones(self.size)).reshape(200, 100)
                        map = splot_grid(np.meshgrid((lat - np.pi / 2) * 180 / np.pi, (long - np.pi) * 180 / np.pi),
                                c, "Homogeneity", llat=llat, ulat=ulat, llon=llon, ulon=ulon, type_plot="contourf",
                                         frame=self.frame, focus=focus)

                    if type_plot == "scatter":
                        c = self.ev_field(xyz, np.ones(self.size))
                        map = splot_scatter((long2 - np.pi) * 180 / np.pi, (lat2 - np.pi / 2) * 180 / np.pi,
                                c, "Homogeneity", llat=llat, ulat=ulat, llon=llon, ulon=ulon, return_=True,
                                        frame=self.frame, focus=focus)



                    if plot_pos:
                        x, y = map((self.nodes_SC[:, 0] - np.pi) * 180 / np.pi,
                                   (self.nodes_SC[:, 1] - np.pi / 2) * 180 / np.pi)
                        map.scatter(x, y, marker="D", color="k")



