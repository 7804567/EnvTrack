import numpy as np
from pycsou.linop.base import SparseLinearOperator, LinOpStack, PyLopLinearOperator
from pycsou.func.loss import SquaredL2Loss
from pycsou.linop.base import KroneckerProduct
from pycsou.core.functional import ProximableFunctional
from pycsou.opt.proxalgs import APGD, PDS
from pandas import DataFrame
import os
import pylops as pp
from scipy.sparse.linalg import LinearOperator
import copy
from splines_code import *

from cdata_code import *

class L1K(ProximableFunctional):

    def __init__(self, K, dim):
        self.K = np.abs(K)
        super().__init__(dim)

    def __call__(self, x):
        return self.K.dot(np.abs(x))

    def prox(self, x, tau):
        return np.sign(x) * np.maximum(np.abs(x) - self.K * tau, 0)

class sfd:

    def __init__(self, data=None, splines=None, T=1, folder_name=None):

        self.op_sparsity_tol = 1e-3
        self.lipschitz_cst_tol = 1e-3
        self.dim = 0
        self.dims= {}
        self.dims_dic = {}
        self.Ops_dic = {}
        self.Evals = {}
        self.weights = {}
        self.normalize = {}

        if folder_name == None:
            self.data = data
            self.T = T
            self.fitted = False
            self.format = list(splines.keys())

            if "space_trend" in self.format and "time_trend" in self.format:
                self.field_psi_trend = splines["space_trend"]
                self.field_zeta_trend = splines["time_trend"]
                self.dims["trend"] = {"space": self.field_psi_trend.size, "time": self.field_zeta_trend.size}

            if "space_seas" in self.format and "time_seas" in self.format:
                self.field_psi_seas = splines["space_seas"]
                self.field_phi_seas = splines["time_seas"]
                self.dims["seas"] = {"space": self.field_psi_seas.size, "time": self.field_phi_seas.size}

        else:
            if isinstance(folder_name, str):
                path = os.path.join(os.getcwd(), folder_name)
                if os.path.exists(path):
                    self.load(folder_name=folder_name)

        if hasattr(self, "field_psi_seas") and hasattr(self, "field_phi_seas"):
            self.dims_dic["seas"] = self.dims["seas"]["space"] * self.dims["seas"]["time"]
            self.op_seas, self.normalize["seas"] = self.get_op_seas()
            self.op_seas.compute_lipschitz_cst()
            self.Ops_dic["seas"] = self.op_seas
            self.dim += self.dims_dic["seas"]
            self.Evals["seas"] = lambda xyz, t, weights: \
                self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
                xyz.dot(self.field_psi_seas.nodes.transpose()), -1, 1))) \
                .dot(weights).dot(self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(
                self.field_phi_seas.nodes.dot(proj_circle(self.T, t).transpose()), -1, 1)))) / self.normalize["seas"]

        if hasattr(self, "field_psi_trend") and hasattr(self, "field_zeta_trend"):

            self.dims_dic["trend"] = self.dims["trend"]["space"] * self.dims["trend"]["time"]
            self.op_trend, self.normalize["trend"] = self.get_op_trend()
            self.op_trend.compute_lipschitz_cst()
            self.Ops_dic["trend"] = self.op_trend
            self.dim += self.dims_dic["trend"]
            self.Evals["trend"] = lambda xyz, t, weights: \
                self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                xyz.dot(self.field_psi_trend.nodes.transpose()), -1, 1))) \
                .dot(weights).dot(self.field_zeta_trend.evaluate(self.field_zeta_trend.proj(
                self.field_zeta_trend.nodes.reshape(self.dims["trend"]["time"], 1) - t.reshape(1, t.shape[0])))) / \
                                                                                self.normalize["trend"]

            if self.field_zeta_trend.kernel:
                self.dims["trendkernel"] = {"space": self.dims["trend"]["space"],
                                            "time": self.field_zeta_trend.dim_kernel}
                self.dims_dic["trendkernel"] = self.dims["trendkernel"]["space"] * self.dims["trendkernel"]["time"]
                self.op_trendkernel, self.normalize["trendkernel"] = self.get_op_trendkernel()
                self.op_trendkernel.compute_lipschitz_cst()
                self.Ops_dic["trendkernel"] = self.op_trendkernel
                self.dim += self.dims_dic["trendkernel"]
                self.Evals["trendkernel"] = lambda xyz, t, weights: \
                    self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                    xyz.dot(self.field_psi_trend.nodes.transpose()), -1, 1))) \
                    .dot(weights / self.normalize["trendkernel"]).dot(self.field_zeta_trend.evaluate_kernel(t))
                for i in range(self.dims["trendkernel"]["time"]):
                    self.Evals["trendkernel"+str(i+1)] = lambda xyz, t, weights: \
                        self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
                            xyz.dot(self.field_psi_trend.nodes.transpose()), -1, 1))) \
                                .dot(weights / self.normalize["trendkernel"][i])[..., None] \
                                .dot(self.field_zeta_trend.evaluate_kernel(t)[i, :][None, ...])

        self.F = self.get_F(code="Full")

        if folder_name is None:
            self.F.compute_lipschitz_cst(tol=self.lipschitz_cst_tol)
        else:
            self.F.lipschitz_cst = np.load(os.path.join(os.path.join(os.getcwd(), folder_name), "info_vector.npy"), \
                                           allow_pickle=True)[1]

    def get_F(self, code):

        if code == "Full":
            code = list(self.Ops_dic.keys())

        list_OP = []
        for c in code:
            list_OP.append(self.Ops_dic[c])

        if len(list_OP) > 1:
            F = LinOpStack(list_OP[0], list_OP[1], axis=1)
            if len(list_OP) > 2:
                for i in range(len(list_OP) - 2):
                    F = LinOpStack(F, list_OP[i + 2], axis=1)

        elif len(list_OP) == 1:
            F = list_OP[0]

        return F

    def get_op_SpaceSeas(self, xyz=None):

        if xyz is None:
            xyz = self.data.xyz

        PSI_seas = self.field_psi_seas.evaluate(self.field_psi_seas.proj(np.clip(
            xyz.dot(self.field_psi_seas.nodes.transpose()), -1, 1)))
        PSI_seas[(PSI_seas < self.op_sparsity_tol)] = 0

        return sp.csr_matrix(PSI_seas), np.max(PSI_seas)

    def get_op_TimeSeas(self, t=None):

        if t is None:
            t = self.data.time

        PHI_seas = self.field_phi_seas.evaluate(self.field_phi_seas.proj(np.clip(
            proj_circle(self.T, t).dot(self.field_phi_seas.nodes.transpose()), -1, 1)))
        PHI_seas[(PHI_seas < self.op_sparsity_tol)] = 0

        return sp.csr_matrix(PHI_seas), np.max(PHI_seas)

    def get_op_SpaceTrend(self, xyz=None):

        if xyz is None:
            xyz = self.data.xyz

        PSI_trend = self.field_psi_trend.evaluate(self.field_psi_trend.proj(np.clip(
            xyz.dot(self.field_psi_trend.nodes.transpose()), -1, 1)))
        PSI_trend[(PSI_trend < self.op_sparsity_tol)] = 0

        return sp.csr_matrix(PSI_trend), np.max(PSI_trend)

    def get_op_TimeTrend(self, t=None):

        if t is None:
            t = self.data.time

        ZETA_trend = self.field_zeta_trend.evaluate(
            self.field_zeta_trend.proj(t[..., None] - self.field_zeta_trend.nodes[None, ...]))
        ZETA_trend[(ZETA_trend < self.op_sparsity_tol)] = 0

        return sp.csr_matrix(ZETA_trend), np.max(ZETA_trend)

    def get_op_TimeTrendKernel(self, t=None):

        if t is None:
            t = self.data.time

        op = self.field_zeta_trend.evaluate_kernel(t).transpose()

        return sp.csr_matrix(op), np.array([1, np.max(op[:, 1])])

    def get_op_seas(self):

        OSS, norm_OSS = self.get_op_SpaceSeas()
        OTS, norm_OTS = self.get_op_TimeSeas()

        def rmatvec(x):

            y = np.zeros((self.dims["seas"]["time"], self.dims["seas"]["space"]))
            for i in range(self.dims["seas"]["time"]):
                y[i, :] = OSS.transpose().dot(OTS[:, i].transpose().multiply(x).toarray().reshape(-1))

            return (y / (norm_OSS * norm_OTS)).reshape(-1)

        def matvec(y):

            x = np.zeros(self.data.true_dim)
            for i in range(self.dims["seas"]["time"]):
                x += OTS[:, i].transpose().multiply(OSS.dot(
                    y[self.dims["seas"]["space"] * i: self.dims["seas"]["space"] * (i + 1)]
                                                )).toarray().reshape(-1)

            return (x / (norm_OSS * norm_OTS))

        return PyLopLinearOperator(pp.LinearOperator(LinearOperator(
            dtype=np.float64,
            shape=(self.data.true_dim, self.dims_dic["seas"]),
            matvec=matvec,
            rmatvec=rmatvec))), (norm_OTS * norm_OSS)

    def get_op_trend(self):

        OST, norm_OST = self.get_op_SpaceTrend()
        OTT, norm_OTT = self.get_op_TimeTrend()


        def rmatvec(x):

            y = np.zeros((self.dims["trend"]["time"], self.dims["trend"]["space"]))
            for i in range(self.dims["trend"]["time"]):
                y[i, :] = OST.transpose().dot(OTT[:, i].transpose().multiply(x).toarray().reshape(-1))

            return (y / (norm_OST* norm_OTT)).reshape(-1)

        def matvec(y):

            x = np.zeros(self.data.true_dim)
            for i in range(self.dims["trend"]["time"]):
                x += OTT[:, i].transpose().multiply(OST.dot(
                    y[self.dims["trend"]["space"] * i: self.dims["trend"]["space"] * (i + 1)]
                                                )).toarray().reshape(-1)

            return (x / (norm_OST* norm_OTT))

        return PyLopLinearOperator(pp.LinearOperator(LinearOperator(
            dtype=np.float64,
            shape=(self.data.true_dim, self.dims_dic["trend"]),
            matvec=matvec,
            rmatvec=rmatvec))), (norm_OTT * norm_OST)

    def get_op_trendkernel(self):

        OST, norm_OST = self.get_op_SpaceTrend()
        OTTK, norm_OTTK = self.get_op_TimeTrendKernel()


        def rmatvec(x):

            y = np.zeros((self.dims["trendkernel"]["time"], self.dims["trendkernel"]["space"]))
            for i in range(self.dims["trendkernel"]["time"]):
                y[i, :] = OST.transpose().dot(OTTK[:, i].transpose().multiply(x).toarray().reshape(-1)) \
                                                                                            / (norm_OST * norm_OTTK[i])

            return y.reshape(-1)

        def matvec(y):

            x = np.zeros(self.data.true_dim)
            for i in range(self.dims["trendkernel"]["time"]):
                x += OTTK[:, i].transpose().multiply(OST.dot(
                    y[self.dims["trendkernel"]["space"] * i: self.dims["trendkernel"]["space"] * (i + 1)]
                                                )).toarray().reshape(-1) / (norm_OST * norm_OTTK[i])
            return x

        return PyLopLinearOperator(pp.LinearOperator(LinearOperator(
            dtype=np.float64,
            shape=(self.data.true_dim, self.dims_dic["trendkernel"]),
            matvec=matvec,
            rmatvec=rmatvec))), norm_OST * norm_OTTK

    def get_Pen(self, lambdas, code="full"):

        if code == "full":
            code = list(self.dims_dic.keys())
        dim = 0
        for c in code:
            dim += self.dims_dic[c]
        d = np.ones(dim)
        format_lambdas = list(lambdas.keys())
        current_indice = 0

        for c in code:
            if c in format_lambdas:
                if lambdas[c] < 0:
                    raise Exception("The lambdas need to be positive.")
                d[current_indice: current_indice + self.dims_dic[c]] *= lambdas[c]
                current_indice += self.dims_dic[c]
            else:
                raise Exception("You need to furnish the method with a lambda for the " + c + " splines.")

        return L1K(d, dim)

    def fit(self, lambdas, max_iter=500, accuracy_threshold=1e-3, method="PDS", x0=None, z0=None):

        l22_loss = SquaredL2Loss(dim=self.data.true_dim, data=self.data.temp) * self.F
        l22_loss.compute_lipschitz_cst()

        if method == "APGD":
            self.fitting_method, key, iterative_method = "APGD", "iterand", APGD

        if method == "PDS":
            self.fitting_method, key, iterative_method = "PDS", "primal_variable", PDS

        iterative_method = iterative_method(F=l22_loss,
                                            G=self.get_Pen(lambdas),
                                            dim=self.dim,
                                            verbose=1,
                                            max_iter=max_iter,
                                            accuracy_threshold=accuracy_threshold,
                                            x0=x0,
                                            z0=z0)

        estimate, converged, diagnostics = iterative_method.iterate()

        self.accuracy_threshold = accuracy_threshold
        self.max_iter = max_iter
        self.converged = converged
        self.diagnostics = diagnostics
        self.fitted = True

        self.fromEstimateToWeights(estimate[key], code="full")
        self.estimate = estimate[key]

    def fromEstimateToWeights(self, estimate, code=None, block_converged=None):
        if code == "full":
            code = list(self.dims_dic.keys())
        current_indice = 0

        if not hasattr(self, "weights"):
            self.weights = {}

        for block in code:
            self.weights[block] = estimate[current_indice: current_indice + self.dims_dic[block]]
            self.weights[block] = self.weights[block].reshape((self.dims[block]["time"],
                                                                self.dims[block]["space"])).transpose()
            current_indice += self.dims_dic[block]

        if "trendkernel" in code:
            for i in range(self.dims["trendkernel"]["time"]):
                self.weights["trendkernel"+str(i+1)] = self.weights["trendkernel"][:, i]

    def evaluation(self, xyz, t, code):

        if isinstance(xyz, np.ndarray) and xyz.shape == (3,):
            eval = np.zeros(t.shape)
        elif isinstance(t, float) or isinstance(t, int) or (isinstance(t, np.ndarray) and t.shape == (1,)):
            eval = np.zeros((xyz.shape[0], 1))
        else:
            eval = np.zeros((xyz.shape[0], t.shape[0]))
        if testreal(t):
            t = np.array([t])

        for c in code:
            eval += self.Evals[c](xyz, t, self.weights[c])

        return eval

    def evaluation2(self, xyz, t, code):
        Creator = {"seas": {"space": self.get_op_SpaceSeas, "time": self.get_op_TimeSeas},
                   "trend": {"space": self.get_op_SpaceTrend, "time": self.get_op_TimeTrend},
                   "trendkernel": {"space": self.get_op_SpaceTrend, "time": self.get_op_TimeTrendKernel}}

        eval = np.zeros(t.shape[0])
        for c in code:
            OS = Creator[c]["space"](xyz)
            OT = Creator[c]["time"](t)
            for i in range(self.dims[c]["time"]):
                eval += OT[:, i].transpose().multiply(OS.dot(\
                    self.weigths[c][self.dims[c]["space"] * i: self.dims[c]["space"] * (i + 1)] )).toarray().reshape(-1)

        return eval

    def load(self, folder_name):

        path = os.path.join(os.getcwd(), folder_name)

        if os.path.exists(path):

            var = np.load(os.path.join(path, "info_vector.npy"), allow_pickle=True)
            self.T = var[0]
            var = np.load(os.path.join(path, "fitted.npy"), allow_pickle=True)
            self.fitted = var[0]
            var = np.load(os.path.join(path, "format.npy"), allow_pickle=True)
            self.format = var

            var = np.load(os.path.join(path, "data.npy"), allow_pickle=True)
            self.data = cdata(var[0], var[1], var[2])

            if "space_seas" in self.format and "time_seas" in self.format:
                path2 = os.path.join(path, "field_psi_seas")
                var = np.load(os.path.join(path2, "float.npy"), allow_pickle=True)
                var2 = np.load(os.path.join(path2, "specification.npy"), allow_pickle=True)
                var3 = np.load(os.path.join(path2, "domain.npy"), allow_pickle=True)
                var4 = np.load(os.path.join(path2, "frame.npy"), allow_pickle=True)
                if isinstance(var3[0], str):
                    var3 = var3[0]
                else:
                    var3 = var3.reshape(-1)
                self.field_psi_seas = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain=var3,
                                                   size=var[1],
                                                   nodes=np.load(os.path.join(path2, "nodes_SC.npy"),
                                                                 allow_pickle=True),
                                                   frame=var4[0])

                path5 = os.path.join(path, "field_phi_seas")
                var = np.load(os.path.join(path5, "float.npy"), allow_pickle=True)
                var2 = np.load(os.path.join(path5, "specification.npy"), allow_pickle=True)
                var3 = np.load(os.path.join(path5, "domain.npy"), allow_pickle=True)
                if isinstance(var3[0], str):
                    var3 = var3[0]
                else:
                    var3 = var3.reshape(-1)
                self.field_phi_seas = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain=var3,
                                                   size=var[1],
                                                   nodes=np.load(os.path.join(path5, "nodes.npy"), allow_pickle=True))

                self.dims["seas"] = {"space": self.field_psi_seas.size, "time": self.field_phi_seas.size}

            if "space_trend" in self.format and "time_trend" in self.format:
                path3 = os.path.join(path, "field_psi_trend")
                var = np.load(os.path.join(path3, "float.npy"), allow_pickle=True)
                var2 = np.load(os.path.join(path3, "specification.npy"), allow_pickle=True)
                var3 = np.load(os.path.join(path3, "domain.npy"), allow_pickle=True)
                var4 = np.load(os.path.join(path3, "frame.npy"), allow_pickle=True)
                if isinstance(var3[0], str):
                    var3 = var3[0]
                else:
                    var3 = var3.reshape(-1)
                self.field_psi_trend = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain=var3,
                                                    size=var[1],
                                                    nodes=np.load(os.path.join(path3, "nodes_SC.npy"),
                                                                  allow_pickle=True),
                                                    frame=var4[0])

                path4 = os.path.join(path, "field_zeta_trend")
                var = np.load(os.path.join(path4, "float.npy"), allow_pickle=True)
                var2 = np.load(os.path.join(path4, "specification.npy"), allow_pickle=True)
                var3 = np.load(os.path.join(path4, "domain.npy"), allow_pickle=True)
                if isinstance(var3[0], str):
                    var3 = var3[0]
                else:
                    var3 = var3.reshape(-1)
                self.field_zeta_trend = spline_field(spline=spline(specification=var2[0], eps=var[0]), domain=var3,
                                                     size=var[1],
                                                     nodes=np.load(os.path.join(path4, "nodes.npy"), allow_pickle=True))

                self.dims["trend"] = {"space": self.field_psi_trend.size, "time": self.field_zeta_trend.size}

            path7 = os.path.join(path, "fit")
            if os.path.exists(path7):
                self.weights = {}
                if hasattr(self, "field_psi_trend") and hasattr(self, "field_zeta_trend"):
                    self.weights["trend"] = np.load(os.path.join(path7, "weights_trend.npy"), allow_pickle=True)
                if hasattr(self, "field_psi_seas") and hasattr(self, "field_phi_seas"):
                    self.weights["seas"] = np.load(os.path.join(path7, "weights_seas.npy"), allow_pickle=True)
                if hasattr(self, "field_psi_trend") and hasattr(self, "field_zeta_trend") \
                        and self.field_zeta_trend.kernel:
                    self.weights["trendkernel"] = np.load(os.path.join(path7, "weights_trendkernel.npy"), \
                                                       allow_pickle=True)
                    for i in range(self.weights["trendkernel"].shape[1]):
                            self.weights["trendkernel" + str(i + 1)] = self.weights["trendkernel"][:, i]
                self.estimate = np.load(os.path.join(path7, "estimate.npy"), allow_pickle=True)

    def save(self, folder_name):

        if not isinstance(folder_name, str):
            raise Exception("folder_name has to be of type str.")
        else:
            path = os.path.join(os.getcwd(), folder_name)

        os.makedirs(path)

        np.save(os.path.join(path, "info_vector"), np.array([self.T, self.F.lipschitz_cst]))
        np.save(os.path.join(path, "format"), np.array([self.format]))
        np.save(os.path.join(path, "fitted"), np.array([self.fitted]))

        if hasattr(self, "field_psi_seas"):
            path1 = os.path.join(path, "field_psi_seas")
            os.makedirs(path1)
            np.save(os.path.join(path1, "float"), np.array([self.field_psi_seas.eps, self.field_psi_seas.size]))
            np.save(os.path.join(path1, "nodes"), self.field_psi_seas.nodes)
            np.save(os.path.join(path1, "nodes_SC"), self.field_psi_seas.nodes_SC)
            np.save(os.path.join(path1, "specification"), np.array([self.field_psi_seas.specification]))
            np.save(os.path.join(path1, "domain"), np.array([self.field_psi_seas.domain]))
            np.save(os.path.join(path1, "frame"), np.array([self.field_psi_seas.frame]))

        if hasattr(self, "field_psi_trend"):
            path2 = os.path.join(path, "field_psi_trend")
            os.makedirs(path2)
            np.save(os.path.join(path2, "float"), np.array([self.field_psi_trend.eps, self.field_psi_trend.size]))
            np.save(os.path.join(path2, "nodes"), self.field_psi_trend.nodes)
            np.save(os.path.join(path2, "nodes_SC"), self.field_psi_trend.nodes_SC)
            np.save(os.path.join(path2, "specification"), np.array([self.field_psi_trend.specification]))
            np.save(os.path.join(path2, "domain"), np.array([self.field_psi_trend.domain]))
            np.save(os.path.join(path2, "frame"), np.array([self.field_psi_trend.frame]))

        if hasattr(self, "field_zeta_trend"):
            path3 = os.path.join(path, "field_zeta_trend")
            os.makedirs(path3)
            np.save(os.path.join(path3, "float"), np.array([self.field_zeta_trend.eps, self.field_zeta_trend.size]))
            np.save(os.path.join(path3, "nodes"), self.field_zeta_trend.nodes)
            np.save(os.path.join(path3, "specification"), np.array([self.field_zeta_trend.specification]))
            np.save(os.path.join(path3, "domain"), np.array([self.field_zeta_trend.domain]))

        if hasattr(self, "field_phi_seas"):
            path4 = os.path.join(path, "field_phi_seas")
            os.makedirs(path4)
            np.save(os.path.join(path4, "float"), np.array([self.field_phi_seas.eps, self.field_phi_seas.size]))
            np.save(os.path.join(path4, "nodes"), self.field_phi_seas.nodes)
            np.save(os.path.join(path4, "specification"), np.array([self.field_phi_seas.specification]))
            np.save(os.path.join(path4, "domain"), np.array([self.field_phi_seas.domain]))

        if self.fitted:

            path6 = os.path.join(path, "fit")
            os.makedirs(path6)
            weights_keys = list(self.weights.keys())

            if "seas" in weights_keys:
                np.save(os.path.join(path6, "weights_seas"), self.weights["seas"])
            if "trend" in weights_keys:
                np.save(os.path.join(path6, "weights_trend"), self.weights["trend"])
            if "trendkernel" in weights_keys:
                np.save(os.path.join(path6, "weights_trendkernel"), self.weights["trendkernel"])
            if hasattr(self, "estimate"):
                np.save(os.path.join(path6, "estimate"), self.estimate)

        np.save(os.path.join(path, "data"), np.array([self.data.year_start, self.data.year_end, self.data.pressure]))

    def CV(self, lambdas, block_size, max_iter=500, accuracy_threshold=1e-3, method="PDS", seed=0):

        rng = np.random.default_rng(seed=seed)
        nblock = (self.data.temp.shape[0] // block_size) + 1
        cv_index = np.repeat(np.arange(0, nblock - 1, 1), block_size)
        cv_index = np.hstack((cv_index, np.ones(self.data.temp.shape[0] - (block_size * (nblock - 1))) * (nblock - 1)))
        cv_index = rng.permutation(cv_index)
        residuals = []

        if method == "APGD":
            self.fitting_method, key, iterative_method = "APGD", "iterand", APGD

        if method == "PDS":
            self.fitting_method, key, iterative_method = "PDS", "primal_variable", PDS

        l22_loss = SquaredL2Loss(dim=self.data.true_dim, data=self.data.temp) * self.F
        l22_loss.compute_lipschitz_cst()
        iterative_method_ = iterative_method(F=l22_loss,
                                            G=self.get_Pen(lambdas),
                                            dim=self.dim,
                                            verbose=1,
                                            max_iter=max_iter,
                                            accuracy_threshold=accuracy_threshold)
        estimate, converged, diagnostics = iterative_method_.iterate()
        self.accuracy_threshold = accuracy_threshold
        self.max_iter = max_iter
        self.converged = converged
        self.diagnostics = diagnostics
        self.fitted = True
        self.fromEstimateToWeights(estimate[key], code="full")
        self.estimate = estimate[key]
        x0 = estimate[key]
        if self.fitting_method == "PDS":
            z0 = estimate["dual_variable"]
        else:
            z0 = None

        for block in range(nblock):
            mask = np.ones(self.data.true_dim, dtype=bool)
            mask[(cv_index == block)] = False
            data = self.data.temp[mask]
            true_dim = data.shape[0]

            def rmatvec(x):
                y = np.zeros(self.F.shape[0])
                y[mask] = x
                return self.F.adjoint(y)

            def matvec(y):
                return self.F(y)[mask]

            F_loc = PyLopLinearOperator(pp.LinearOperator(LinearOperator(
                dtype=np.float64, shape=(true_dim, self.F.shape[1]), matvec=matvec, rmatvec=rmatvec)))
            F_loc.lipschitz_cst = self.F.lipschitz_cst
            F_loc.diff_lipschitz_cst = self.F.diff_lipschitz_cst
            l22_loss_ = SquaredL2Loss(dim=true_dim, data=data) * F_loc
            l22_loss_.lipschitz_cst = l22_loss.lipschitz_cst
            iterative_method_ = iterative_method(F=l22_loss_,
                                                G=self.get_Pen(lambdas),
                                                dim=self.dim,
                                                verbose=1,
                                                max_iter=max_iter,
                                                accuracy_threshold=accuracy_threshold,
                                                x0=x0,
                                                z0=z0)
            estimate, converged, diagnostics = iterative_method_.iterate()
            residuals.append((self.data.temp - self.F(estimate[key]))[~mask])
            print("CV: "+str(len(residuals))+"/"+str(nblock)+" done.")

        return cv_index, residuals