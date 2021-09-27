import h5py
import torch
import json

import numpy as np


class BFM:
    def __init__(self, path):
        self._shape_mu = None
        self._shape_pcab = None
        self._shape_pca_var = None
        self._faces = None
        self._expression_mu = None
        self._expression_pcab = None
        self._expression_pca_var =  None

    @property
    def shape_mu(self):
        return np.array(self._shape_mu)

    @property
    def shape_pcab(self):
        return np.array(self._shape_pcab)

    @property
    def shape_pca_var(self):
        return np.array(self._shape_pca_var)

    @property
    def expression_mu(self):
        return np.array(self._expression_mu)

    @property
    def expression_pcab(self):
        return np.array(self._expression_pcab)

    @property
    def expression_pca_var(self):
        return np.array(self._expression_pca_var)

    @property
    def faces(self):
        return np.array(self._faces)

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1)
        Returns:
            vertices: (nver, 3)
        '''
        raise NotImplementedError("Implement in subclass")

    def get_mean(self):
        return self.shape_mu.reshape(-1, 3)

    def apply_Rts(self, verts, R, t, s):
        transformed_verts  = s * np.matmul(verts, R) + t
        return transformed_verts


class BFM2017(BFM):
    N_COEFF_SHAPE = 199
    N_COEFF_EXPRESSION = 100

    def  __init__(self, path_model, path_uv=None):
        super().__init__(path_model)
        import h5py
        self.file = h5py.File(path_model, 'r')
        self._shape_mu = self.file["shape"]["model"]["mean"]
        self._shape_pcab = self.file["shape"]["model"]["pcaBasis"]
        self._shape_pca_var = self.file["shape"]["model"]["pcaVariance"]
        self._faces = np.transpose(self.file["shape"]["representer"]["cells"])
        self._expression_mu = self.file["expression"]["model"]["mean"]
        self._expression_pcab = self.file["expression"]["model"]["pcaBasis"]
        self._expression_pca_var =  self.file["expression"]["model"]["pcaVariance"]

        if "model2017-1_face12_nomouth.h5" in path_model:
            # Model without ears and throat
            self.N_VERTICES = 28588
            self.N_FACES = 56572
        else:
            # Model with ears and throat
            self.N_VERTICES = 53149
            self.N_FACES = 105694

        assert 0 == self.faces.min()
        assert self.faces.max() == self.N_VERTICES - 1

        if path_uv is not None:
            self.vertices_uv, self.faces_uv = self.load_uv(path_uv)

    def load_uv(self, path_uv):
        with open(path_uv, 'r') as f:
            uv_para = json.load(f)
        verts_uvs = np.array(uv_para['textureMapping']['pointData'])
        faces_uvs = np.array(uv_para['textureMapping']['triangles'])
        return verts_uvs, faces_uvs

    def __delete__(self, instance):
        self.file.close()

    def generate_vertices(self, shape_para, exp_para):
        if shape_para.shape != (self.N_COEFF_SHAPE, 1):
            shape_para = shape_para.reshape(self.N_COEFF_SHAPE, 1)
        if exp_para.shape != (self.N_COEFF_EXPRESSION, 1):
            exp_para = exp_para.reshape(self.N_COEFF_EXPRESSION, 1)
        vertices = self.shape_mu.reshape(-1, 1) + self.shape_pcab @ shape_para + self.expression_pcab @ exp_para
        vertices = np.reshape(vertices, [int(3), int(self.N_VERTICES)], 'F').T
        return vertices.astype(np.float32)

    def sample(self, n, std_multiplier, variance):
        std = np.sqrt(variance) * std_multiplier
        std = np.broadcast_to(std, (n, std.shape[0]))
        mu = np.zeros_like(std)
        samples = np.random.normal(mu, std)
        return samples

    def sample_shape(self, n, std_multiplier=1):
        return self.sample(n, std_multiplier, self.shape_pca_var)

    def sample_expression(self, n, std_multiplier=1):
        return self.sample(n, std_multiplier, self.expression_pca_var)


class BFM2017Tensor:
    N_VERTICES = 28588
    N_FACES = 56572
    N_COEFF_SHAPE = 199
    N_COEFF_EXPRESSION = 100

    def __init__(self, path_model, path_uv=None, device='cuda', verbose=False):
        print("Loading BFM 2017 into GPU... (this can take a while)")
        self.device = device
        self.file = h5py.File(path_model, 'r')
        # This can take a few seconds
        self.shape_mu = torch.Tensor(self.file["shape"]["model"]["mean"]).reshape(self.N_VERTICES*3, 1).to(device).float()
        self.shape_pcab = torch.Tensor(self.file["shape"]["model"]["pcaBasis"]).reshape(self.N_VERTICES*3, self.N_COEFF_SHAPE).to(device).float()
        self.shape_pca_var = torch.Tensor(self.file["shape"]["model"]["pcaVariance"]).reshape(self.N_COEFF_SHAPE).to(device).float()
        self.expression_pcab = torch.Tensor(self.file["expression"]["model"]["pcaBasis"]).reshape(self.N_VERTICES*3, self.N_COEFF_EXPRESSION).to(device).float()
        self.expression_pca_var = torch.Tensor(self.file["expression"]["model"]["pcaVariance"]).reshape(self.N_COEFF_EXPRESSION).to(device).float()
        self.faces = torch.Tensor(np.transpose(self.file["shape"]["representer"]["cells"])).reshape(self.N_FACES, 3).to(device)

        if path_uv is not None:
            self.vertices_uv, self.faces_uv = self.load_uv(path_uv)
        print("Done")

    def load_uv(self, path_uv):
        with open(path_uv, 'r') as f:
            uv_para = json.load(f)
        verts_uvs = torch.Tensor(uv_para['textureMapping']['pointData']).reshape(-1, 3).to(self.device)
        faces_uvs = torch.Tensor(np.array(uv_para['textureMapping']['triangles'])).reshape(-1, 3).to(self.device)
        return verts_uvs, faces_uvs

    def __delete__(self, instance):
        self.file.close()

    def generate_vertices(self, shape_para, exp_para):
        if shape_para.shape != (self.N_COEFF_SHAPE, 1):
            shape_para = shape_para.reshape(self.N_COEFF_SHAPE, 1)
        if exp_para.shape != (self.N_COEFF_EXPRESSION, 1):
            exp_para = exp_para.reshape(self.N_COEFF_EXPRESSION, 1)
        vertices = self.shape_mu +\
                   self.shape_pcab @ shape_para + \
                   self.expression_pcab @ exp_para
        return vertices.reshape(-1, 3).float()

    def sample(self, n, std_multiplier, variance):
        std = torch.sqrt(variance) * std_multiplier
        std = std.expand((n, std.shape[0]))
        q = torch.distributions.Normal(torch.zeros_like(std).to(std.device), std * std_multiplier)
        samples = q.rsample()
        return samples

    def sample_shape(self, n, std_multiplier=1):
        return self.sample(n, std_multiplier, self.shape_pca_var)

    def sample_expression(self, n, std_multiplier=1):
        return self.sample(n, std_multiplier, self.expression_pca_var)
