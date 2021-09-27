import json
from typing import Union, List, Tuple

import numpy as np
import torch
import torchvision
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV
)

from mutil.pytorch_utils import to_tensor


class Renderer(torch.nn.Module):
    def __init__(self, dist=2, elev=0, azimuth=180, fov=40, image_size=256, R=None, T=None, cameras=None, return_format="torch", device='cuda'):
        super().__init__()
        # If you provide R and T, you don't need dist, elev, azimuth, fov
        self.device = device
        self.return_format = return_format

        # Data structures and functions for rendering
        if cameras is None:
            if R is None and T is None:
                R, T = look_at_view_transform(dist, elev, azimuth)
            cameras = FoVPerspectiveCameras(R=R, T=T, znear=1, zfar=10000, fov=fov, degrees=True, device=device)
            # cameras = PerspectiveCameras(R=R, T=T, focal_length=1.6319*10, device=device)

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,  # no blur
            bin_size=0,
        )
        # Place lights at the same point as the camera
        location = T
        if location is None:
            location = ((0,0,0),)
        lights = PointLights(ambient_color=((0.3, 0.3, 0.3),), diffuse_color=((0.7, 0.7, 0.7),), device=device,
                             location=location)

        self.mesh_rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            )
        self._renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
        self.cameras = self.mesh_rasterizer.cameras

    def _flatten(self, a):
        return torch.from_numpy(np.array(a)).reshape(-1, 1).to(self.device)

    def format_output(self, image_tensor, return_format=None):
        if return_format is None:
            return_format = self.return_format

        if return_format == "torch":
            return image_tensor
        elif return_format == "pil":
            if len(image_tensor.shape) == 4:
                vis = [self.format_output(t, return_format) for t in image_tensor]
                return vis
            else:
                to_pil = torchvision.transforms.ToPILImage()
                vis = image_tensor.detach().cpu()
                return to_pil(vis)
        elif return_format == "np_raw":
            return image_tensor.detach().cpu().permute(0,2,3,1).numpy()
        elif return_format == "np":
            pil_image = self.format_output(image_tensor, return_format='pil')
            if isinstance(pil_image, list):
                pil_image = [np.array(img) for img in pil_image]
            return np.array(pil_image)


class ImagelessTexturesUV(TexturesUV):
    def __init__(self,
        faces_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        verts_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        padding_mode: str = "border",
        align_corners: bool = True,
    ):
        self.device = faces_uvs[0].device
        batch_size = faces_uvs.shape[0]
        maps = torch.zeros(batch_size, 2, 2, 3).to(self.device)  # This is simply to instantiate a texture, but it is not used.
        super().__init__(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs, padding_mode=padding_mode, align_corners=align_corners)

    def sample_pixel_uvs(self, fragments, **kwargs) -> torch.Tensor:
        """
        Copied from super().sample_textures and adapted to output pixel_uvs instead of the sampled texture.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordianates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """
        if self.isempty():
            faces_verts_uvs = torch.zeros(
                (self._N, 3, 2), dtype=torch.float32, device=self.device
            )
        else:
            packing_list = [
                    i[j] for i, j in zip(self.verts_uvs_list(), self.faces_uvs_list())
                ]
            faces_verts_uvs = torch.cat(packing_list)
            # Each vertex yields 3 triangles with u,v coordinates (N, 3, 2)
        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape
        # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)
        pixel_uvs = pixel_uvs * 2.0 - 1.0
        return pixel_uvs


class UVRenderer(Renderer):
    def __init__(self, verts_uv, faces_uv, dist=2, elev=0, azimuth=180, fov=40, image_size=256, R=None, T=None, cameras=None):
        # obj path should be the path to the obj file with the UV parametrization
        super().__init__(dist=dist, elev=elev, azimuth=azimuth, fov=fov, image_size=image_size, R=R, T=T, cameras=cameras)
        self.verts_uvs = verts_uv
        self.faces_uvs = faces_uv

    def render(self, meshes):
        batch_size = len(meshes)
        texture = ImagelessTexturesUV(verts_uvs=self.verts_uvs.expand(batch_size, -1, -1), faces_uvs=self.faces_uvs.expand(batch_size, -1, -1))
        # Currently only supports one mesh in meshes
        fragments = self.mesh_rasterizer(meshes)
        rendered_uv = texture.sample_pixel_uvs(fragments)
        return self.format_output(rendered_uv)


class BFMUVRenderer(Renderer):
    def __init__(self, json_path, *args, **kwargs):
        # json_path = ".../face12.json"
        super().__init__(*args, **kwargs)

        with open(json_path, 'r') as f:
            uv_para = json.load(f)
        verts_uvs = np.array(uv_para['textureMapping']['pointData'])
        faces_uvs = np.array(uv_para['textureMapping']['triangles'])

        verts_uvs = to_tensor(verts_uvs).unsqueeze(0).float()
        faces_uvs = to_tensor(faces_uvs).unsqueeze(0).long()
        self.texture = ImagelessTexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs).to(self.device)

    def render(self, meshes):
        # Currently only supports one mesh in meshes
        fragments = self.mesh_rasterizer(meshes)
        rendered_uv = self.texture.sample_pixel_uvs(fragments)
        rendered_uv = rendered_uv.permute(0, 3, 1, 2)  # to CHW
        return self.format_output(rendered_uv)

