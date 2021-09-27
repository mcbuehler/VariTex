from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform


class FFHQ:
    # We load the inital images with this size. It should be the same as the predicted segmentation masks.
    # The original FFHQ images have resolution 1024x1024.
    image_width = 512
    image_height = 512

    transform_params = dict(
        degrees=15,
        translate=(0.2, 0.2),
        scale=(1, 1.2),
        flip_p=0.5,
    )
    transform_params_light = dict(
        degrees=5,
        translate=(0.1, 0.1),
        scale=(1, 1.2),
        flip_p=0.5,
    )

    @classmethod
    def get_transform_params(cls, mode):
        keys = []
        if mode == "all":
            keys = cls.transform_params.keys()
        else:
            if "d" in mode:
                keys.append("degrees")
            if "t" in mode:
                keys.append("translate")
            if "s" in mode:
                keys.append("scale")
            if "f" in mode:
                keys.append("flip_p")
        params = {k: v for k, v in cls.transform_params.items() if k in keys}
        return params


class Camera:
    @staticmethod
    def get_camera():
        # Camera is at the origin, looking at the negative z axis.
        R, T = look_at_view_transform(eye=((0, 0, 0),), at=((0, 0, -1),), up=((0, 1, 0),))
        cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T, fov=30, zfar=1000)
        return cameras
