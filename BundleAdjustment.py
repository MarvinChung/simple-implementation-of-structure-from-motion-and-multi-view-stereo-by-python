#based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
#add principle point

from scipy.sparse import lil_matrix
import numpy as np
from scipy.optimize import least_squares

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    #px,py: princple points in pixels
    #f: focal length in pixels
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    px = camera_params[:, 9]
    py = camera_params[:, 10]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    points_proj += np.concatenate((px.reshape(-1,1),py.reshape(-1,1)),axis=1)
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 11].reshape((n_cameras, 11))
    points_3d = params[n_cameras * 11:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
#     print("round")
#     for idx,i,j in zip(point_indices[:5],points_proj[:5], points_2d[:5]):
#         print(points_3d[idx])
#         print("project 3d:",i)
#         print("2d:",j)
#     input("")
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 11 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(11):
        A[2 * i, camera_indices * 11 + s] = 1
        A[2 * i + 1, camera_indices * 11 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 11 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 11 + point_indices * 3 + s] = 1

    return A