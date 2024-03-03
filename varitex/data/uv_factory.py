import json

import numpy as np
import torch
from mutil.bfm2017 import BFM2017Tensor, BFM2017
from mutil.pytorch_utils import to_tensor, theta2rotation_matrix
from mutil.renderer import UVRenderer
from mutil.threed_utils import apply_Rts
from pytorch3d.structures import Meshes

from varitex.data.dataset_specifics import Camera
from varitex.modules.custom_module import CustomModule


class BFMUVFactory(CustomModule):
    """
        facemodels_path = ...
        path_bfm = os.path.join(facemodels_path, "basel_facemodel/model2017-1_face12_nomouth.h5")
        path_uv = os.path.join(facemodels_path, "basel_facemodel/face12.json")
        factory = BFMUVFactory(path_bfm, path_uv, image_size=256)
    """

    def __init__(self, opt, use_bfm_gpu=False):
        super().__init__(opt)
        self.to(opt.device)

        # This factory uses the Basel Face Model
        self.use_bfm_gpu = use_bfm_gpu  # Loads the BFM into GPU for faster mesh generation
        if self.use_bfm_gpu:
            self.bfm = BFM2017Tensor(self.opt.path_bfm, device=opt.device)
        else:
            self.bfm = BFM2017(self.opt.path_bfm, self.opt.path_uv)
        self.faces_tensor = to_tensor(self.bfm.faces, self.device).unsqueeze(0)

        verts_uv, faces_uv = self.load_verts_faces_uv(opt.path_uv)
        self.uv_renderer = UVRenderer(verts_uv, faces_uv, image_size=self.opt.image_h, cameras=Camera.get_camera())

    def unsqueeze_transform(self, R, t, s):
        if len(R.shape) == 2:
            R, t, s = R.unsqueeze(0), t.unsqueeze(0), s.unsqueeze(0)
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        R, t, s = R.float().to(self.device), t.float().to(self.device), s.float().to(self.device)
        return R, t, s

    def load_verts_faces_uv(self, json_path):
        with open(json_path, 'r') as f:
            uv_para = json.load(f)
        verts_uvs = np.array(uv_para['textureMapping']['pointData'])
        faces_uvs = np.array(uv_para['textureMapping']['triangles'])

        verts_uvs = to_tensor(verts_uvs, self.device, dtype=float).unsqueeze(0).float()
        faces_uvs = to_tensor(faces_uvs, self.device, dtype=float).unsqueeze(0).long()

        return verts_uvs, faces_uvs

    def generate_vertices(self, sp, ep):
        # sp, ep need to have a batch dimension
        batch_size = ep.shape[0]
        if self.use_bfm_gpu:
            vertices = [self.bfm.generate_vertices(sp[i], ep[i]) for i in range(batch_size)]
        else:
            vertices = [self.bfm.generate_vertices(sp[i].detach().cpu().numpy(), ep[i].detach().cpu().numpy()) for i in
                        range(batch_size)]
            vertices = [to_tensor(v, self.device) for v in vertices]
        vertices = torch.stack(vertices)
        return vertices

    def get_posed_meshes(self, sp, ep, R, t, s, batch_size, correct_translation):
        # Generate the vertices
        vertices = self.generate_vertices(sp, ep)
        # Match the batch size
        faces = self.faces_tensor.expand(batch_size, self.bfm.N_FACES, 3)
        vertices = apply_Rts(vertices, R, t, s, correct_tanslation=correct_translation)
        meshes = Meshes(verts=vertices, faces=faces)
        return meshes

    def getUV(self, R=torch.eye(3).unsqueeze(0), t=torch.Tensor((0, 0, -35)).unsqueeze(0),
              s=torch.Tensor((0.1,)).unsqueeze(0), sp=torch.zeros((199,)).unsqueeze(0),
              ep=torch.zeros((100,)).unsqueeze(0), correct_translation=True):
        R, t, s = self.unsqueeze_transform(R, t, s)
        assert len(sp.shape) == 2 and sp.shape[-1] == 199, "Should come in shape (batch_size, 199), but is {}".format(
            sp.shape)
        assert len(ep.shape) == 2 and ep.shape[-1] == 100, "Should come in shape (batch_size, 100), but is {}".format(
            ep.shape)
        batch_size = R.shape[0]
        meshes = self.get_posed_meshes(sp, ep, R, t, s, batch_size, correct_translation)
        uv = self.uv_renderer.render(meshes)
        return uv

    def get_sampled_uvs_shape(self, n, std_multiplier=1, ep=np.zeros(100, )):
        shapes = self.bfm.sample_shape(n, std_multiplier)
        uv_list = [self.getUV(sp, ep) for sp in shapes]
        return uv_list

    def get_sampled_uvs_expression(self, n, std_multiplier=1, sp=np.zeros(199, )):
        expressions = self.bfm.sample_expression(n, std_multiplier)
        uv_list = [self.getUV(sp, ep) for ep in expressions]
        return uv_list

    def get_posed_uvs(self, sp=torch.zeros((199,)), ep=torch.zeros((100,)), deg_range=torch.arange(-30, 31, 15),
                      t=torch.Tensor((0, 0, -35)), s=torch.Tensor((0.1,))):
        uv_list = list()
        for theta_y in deg_range:
            for theta_x in deg_range:
                theta = theta_x, theta_y, 0
                R = theta2rotation_matrix(theta_all=theta)
                uv = self.getUV(sp, ep, R, t, s)
                uv_list.append(uv)
        return uv_list
