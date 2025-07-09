import numpy as np
import toml
import pandas as pd
import cv2

def plane_from_points(P1, P2, P3):
    # Convert to numpy arrays
    P1, P2, P3 = map(np.array, (P1, P2, P3))
    
    # Two vectors in the plane
    v1 = P2 - P1
    v2 = P3 - P1

    # Normal vector
    normal = np.cross(v1, v2)
    A, B, C = normal

    # Plane equation: Ax + By + Cz + D = 0
    D = -np.dot(normal, P1)

    return A, B, C, D  # coefficients of the plane

def line_from_points(P1, P2):
    # Convert to numpy arrays
    P1, P2 = map(np.array, (P1, P2))
    
    # Two vectors in the plane
    v1 = P2 - P1
    A, B, C = v1

    return A, B, C  # coefficients of the line

def triangulate_dlt_vectorized(
    points: np.ndarray,
    projection_matrices: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Triangulate 3D points from multiple camera views using Direct Linear Transform.

    Args:
        points: Array of N 2D normalized (undistorted) points from each camera view M of
            dtype float64 and shape (M, N, 2) where N is the number of points.
        projection_matrices: Array of (3, 4) projection matrices for each camera M of
            shape (M, 3, 4).

    Returns:
        Triangulated 3D points of shape (N, 3) where N is the number of points.
    """
    # Get A of shape (N, 2M, 4) s.t. each 3D point has A of shape 2M x 4
    a = get_dlt_transformation(points, projection_matrices)
    # Remove rows with NaNs before SVD which may result in a ragged A (hence for loop)
    points_3d = []
    for a_slice in a:
        # Check that we have at least 2 views worth of non-nan points.
        nan_mask = np.isnan(a_slice)  # 2M x 4
        has_enough_matches = np.all(~nan_mask, axis=1).sum() >= 4  # Need 2 (x, y) pairs

        point_3d = np.full(3, np.nan)
        if has_enough_matches:
            a_no_nan = a_slice[~nan_mask].reshape(-1, 4, order="C")
            _, _, vh = np.linalg.svd(a_no_nan)
            point_3d = vh[-1, :-1] / vh[-1, -1]

        points_3d.append(point_3d)

    points_3d = np.array(points_3d)

    return points_3d

def load_calibration_data(file_path):
    with open(file_path, "r") as f:
        calib_data = toml.load(f)
    return calib_data

def load_annotations(file_path):
    id_type_map = {
        1: "ground",
        2: "origin",
        3: "wall"
    }
    df = pd.read_csv(file_path)
    df['type'] = df['point_type'].map(id_type_map)
    df['frame'] = df['frame'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    return df

def map_vid_cam_data(calib_data, annotations_df):
    dict_vid_cam_data = {}
    for vid_name in annotations_df["video"].unique():
        vid_name = vid_name.split(".")[0]
        for cam_name, cam_data in calib_data.items():
            if vid_name == cam_data["name"]:
                dict_vid_cam_data[vid_name] = calib_data[cam_name]
    return dict_vid_cam_data

def undistort_points(distorted_points, cam_calib_data) -> np.ndarray:
    """Takes in 2D points and calibration data for a single camera, and returns the undistorted points.
    """
    K = np.array(cam_calib_data['matrix'])
    dist_coeffs = np.array(cam_calib_data['distortions'])
    height, width = cam_calib_data['size']
    undistorted_points = cv2.undistortPoints(
        distorted_points,
        K, 
        dist_coeffs, 
    )

    return undistorted_points

def compute_global_coordinate_frame(calib_data, imgs, annotations_path):
    """Takes in calibration toml for all cams, list of images with their corresponding 
    ground/origin/wall annotations in a dataframe, and returns the global reference frame.
    """
    # load annotated points on global reference frame
    df = load_annotations(annotations_path)
    # map from video name in annotations, to cam calibration data
    dict_vid_calib_data = map_vid_cam_data(calib_data, annotations_df)

    # convert to dict of 2D points in each view
    dict_distorted_points = {}
    for vid_name, data in df.groupby("video"):
        vid_name = vid_name.split(".")[0]
        cam_calib_data = dict_vid_calib_data[vid_name]
        points = data.loc[:, ["x", "y"]].values
        # undistort the annotated points
        dict_undistorted_points = undistort_points(points, cam_calib_data)
        dict_distorted_points[vid_name] = points
    
    # triangulate all points


    # Get a vector in the ground plane in 3d

    # Get the z axis vector (origin to wall) in 3d

    # Cross product to get the 3rd coordinate axis

    # return a rotation matrix; each col is an axis of the global coordinate frame

    # translation vector is just the 3d location of the origin point in the arbitrary ref plane
    # each instance keypoint can simply be translated by its distance to this 3d point in the arbitrary frame