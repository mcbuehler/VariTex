import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# add path for demo utils functions
import numpy as np
import torch


def setup_plots():
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")



def align_ls(mp_verts, bfm_verts, idx_mp, idx_bfm):
    tmp_mp = mp_verts.clone()[idx_mp].reshape(1, -1, 3).float()
    tmp_bfm = bfm_verts.clone()[idx_bfm].reshape(1, -1, 3).float()
    R, t, s = corresponding_points_alignment(tmp_mp, tmp_bfm, estimate_scale=True)
    mp_verts_aligned = s * mp_verts.mm(R[0]) + t
    return mp_verts_aligned, R, t, s


def unwrap(vertices, faces):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)  # Set process to False, o.w. will change vertex order
    mesh = mesh.unwrap()  # This will change the vertex order
    uv = mesh.visual.uv
    return uv


def plot_mesh(verts, faces, show=False):
    import plotly.graph_objects as go
    from mutil.pytorch_utils import tensor2np

    verts = tensor2np((verts))
    # faces = tensor2np(faces)
    x, y, z = verts[:,0], verts[:,1],\
              verts[:,2]
    # points3d(x,y,z)
    # i, j, k = faces[:,0], faces[:,1], faces[:,2]
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, opacity=0.9)])
    if show:
        plt.show()
    return fig


def apply_Rts(vertices, R, t, s, correct_tanslation=False):
    assert len(vertices.shape) == 3, "Needs batch size"
    bs, n_vertices = vertices.shape[:2]
    s = s.view(bs, 1, 1).expand(bs, n_vertices, 3)
    t = t.view(bs, 1, 3).expand(bs, n_vertices, 3)
    # print('verts mean', vertices.mean(1))
    if correct_tanslation:
        # This is the center point that we rotate around
        center = vertices.mean(-2).unsqueeze(1).expand(vertices.shape)
        rotated_vertices = (vertices - center).matmul(R) + center
    else:
        rotated_vertices = vertices.matmul(R)
    # print('rotated', rotated_vertices.mean(1))
    transformed_vertices = s * rotated_vertices + t
    # print('transformed', transformed_vertices.mean(1))
    return transformed_vertices


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R, eps=1e-6):
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n >= eps:
        raise Warning("Warning: Rt != R. delta norm: {}".format(n))
    return n < eps


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R, eps=1e-6, degrees=False):
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    assert isRotationMatrix(R, eps)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    result = np.array([x, y, z])
    if degrees:
        result = np.rad2deg(result)
    return result


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    # Angles in radians plz
    # theta order: x, y z
    # pitch, yaw, roll
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def view_matrix_fps(pitch, yaw, eye, project_eye=False):
    """
    Following https://www.3dgep.com/understanding-the-view-matrix/#The_View_Matrix
    Right-handed coordinate system
    if project_eye: the eye will be dotted with the axis, otherwise not.

    """
    sp = np.sin(pitch)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    cy = np.cos(yaw)
    x_axis = [cy, 0, -sy]
    y_axis = [sy * sp, cp, cy * sp]
    z_axis = [sy * cp, -sp, cp * cy]
    if project_eye:
        translation = [-np.dot(x_axis, eye), -np.dot(y_axis, eye), -np.dot(z_axis, eye)]
    else:
        translation = eye
    V = np.array([x_axis, y_axis, z_axis, translation])  # 4x3
    V = V.T  # 3x4
    V = np.vstack((V, np.array([0, 0, 0, 1.0]).reshape(1, 4)))
    return V


def fov2focal(fov, image_size, degree=True):
    """
    Computes focal length from similarity of triangles
    Args:
        fov: angle of view in degrees
        image_size: in pixels or any metric unit
        degree: True if fov is in degrees, else false

    Returns: focal length in the same unit as image_size

    """
    A = image_size / 2.0  # Half the image size
    a = fov / 2.0  # Half the fov angle
    if degree:
        a = np.deg2rad(a)
    f = A / np.tan(a)  # numpy expects angles in radians
    return f


def create_cam2world_matrix(forward_vector, origin):
    # Adapted from pigan repo
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    def normalize_vec(v):
        return v / np.linalg.norm(v)
    forward_vector = normalize_vec(forward_vector)
    up_vector = np.array([0, 1.0, 0])

    left_vector = normalize_vec(np.cross(up_vector, forward_vector, axis=-1))

    up_vector = normalize_vec(np.cross(forward_vector, left_vector, axis=-1))

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def project_mesh(vertices, img, cam, color=[255, 0, 0]):
    """

    Args:
        vertices: Nx3 np.array
        img: PIL.Image (np.array also works)
        cam: K, P: K is the 3x3 camera matrix (maps to camera space; focal, princiapl point, skew), P is the 4x4 matrix (R|T) and maps to the vertex space (will be inverted)
        color:

    Returns: PIL.Image with projected vertices

    """
    from PIL import Image
    # Adapted from https://github.com/CrisalixSA/h3ds

    # Expand cam attributes
    K, P = cam  # K is the 3x3 camera matrix (focal, principal point, skew), P is the 4x4 projection matrix (R|T)
    # P projects from camera to world space, so we need the inverse
    P_inv = np.linalg.inv(P)

    # Project mesh vertices into 2D
    p3d_h = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) # Homogeneous
    p2d_h = (K @ P_inv[:3, :] @ p3d_h.T).T  # Apply the camera and the world^-1 projection. Transpose result
    p2d = p2d_h[:, :-1] / p2d_h[:, -1:]  # Divide by Z to project to image plane
    # The p2d now contains the indices of the values as pixels

    # Draw p2d to image
    img_proj = np.array(img)
    p2d = np.clip(p2d, 0, img.width - 1).astype(np.uint32)  # Discretize them
    img_proj[p2d[:, 1], p2d[:, 0]] = color

    return Image.fromarray(img_proj.astype(np.uint8))


def points2homo(points, times=1):
    if len(points.shape) == 2:
        points_h = np.hstack((points, np.ones((points.shape[0], times))))  # N,4
    elif len(points.shape) == 3:
        # Has a batch dimension
        points_h = np.dstack((points, np.ones((points.shape[0], points.shape[1], times))))
    else:
        raise Warning("Invalid shape")
    return points_h


def scaling_transform(s):
    scale_transform = np.zeros((4,4))
    scale_transform[0,0] = s
    scale_transform[1,1] = s
    scale_transform[2,2] = s
    scale_transform[3,3] = 1
    return scale_transform