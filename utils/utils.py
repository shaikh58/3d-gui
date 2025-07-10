import numpy as np
import toml
import pandas as pd
import cv2

def fit_plane(points):
    centroid = points.mean(axis=0)
    U, S, Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]  # normal to the plane is the last singular vector
    return normal / np.linalg.norm(normal)

def get_dlt_transformation(
    points: np.ndarray,
    projection_matrices: np.ndarray,
) -> np.ndarray:
    """Compute the transformation A used in Direct Linear Transform AX = 0.

    Args:
        points: Array of N 2D points from each camera view M of dtype float64 and shape
            (M, N, 2) where N is the number of points.
        projection_matrices: Array of (3, 4) projection matrices for each camera M of
            shape (M, 3, 4).

    Returns:
        The transfomation matrix A of shape (N, 2M, 4) where N is the number of
        points and M is the number of cameras.
    """
    n_cameras, n_points, _ = points.shape

    # Flatten points to shape needed for multiplication
    points_flattened = points.reshape(n_cameras, 2 * n_points, 1, order="C")

    # Create row selector matrix to select correct rows from projection matrix
    row_selector = np.zeros((n_cameras * n_points, 2, 2))
    row_selector[:, 0, 0] = -1  # Negate 1st row of projection matrix for x
    row_selector[:, 1, 1] = -1  # Negate 2nd row of projection matrix for y
    row_selector = row_selector.reshape(n_cameras, 2 * n_points, 2, order="C")

    # Concatenate row selector and points matrices to shape (M, 2N, 3)
    left_matrix = np.concatenate((row_selector, points_flattened), axis=2)

    # Get A (stacked in a weird way) of shape (M, 2N, 4)
    a_stacked: np.ndarray = np.matmul(left_matrix, projection_matrices)

    # Reorganize A to shape (N, 2M, 4) s.t. each 3D point has A of shape 2M x 4
    a = (
        a_stacked.reshape(n_cameras, n_points, 2, 4)
        .transpose(1, 0, 2, 3)
        .reshape(n_points, 2 * n_cameras, 4)
    )

    return a

def triangulate_dlt_vectorized(
    points: np.ndarray,
    projection_matrices: np.ndarray,
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

def axangle_to_rot_mat(axangle):
    """Convert axis-angle representation to rotation matrix.
    """
    return cv2.Rodrigues(axangle)[0]

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
    df['video'] = df['video'].str.split(".").str[0]
    return df

def map_vid_cam_data(calib_data, annotations_df):
    dict_vid_cam_data = {}
    for vid_name in annotations_df["video"].unique():
        for cam_name, cam_data in calib_data.items():
            if "cam" in cam_name:
                if vid_name == cam_data["name"]:
                    dict_vid_cam_data[vid_name] = calib_data[cam_name]
    return dict_vid_cam_data

def undistort_points(distorted_points, cam_calib_data) -> np.ndarray:
    """Takes in 2D points and calibration data for a single camera, and returns the undistorted points.
    """
    K = np.array(cam_calib_data['matrix'])
    dist_coeffs = np.array(cam_calib_data['distortions'])
    undistorted_points = cv2.undistortPoints(
        distorted_points,
        K, 
        dist_coeffs, 
    )

    return undistorted_points

def compute_global_coordinate_frame(calib_data_path, annotations_path):
    """Takes in calibration toml path, and ground/origin/wall annotations 
    in a csv file, and returns the global reference frame.
    Assumes point type 1=ground, 2=origin, 3=wall.
    """
    # load calibration data
    calib_data = load_calibration_data(calib_data_path)
    # load annotated points on global reference frame
    annotations_df = load_annotations(annotations_path)
    # map from video name in annotations, to cam calibration data
    dict_vid_calib_data = map_vid_cam_data(calib_data, annotations_df)

    # for each view, undistort the annotated points
    list_undistorted_points = []
    for vid_name, data in annotations_df.groupby("video"):
        vid_name = vid_name.split(".")[0]
        cam_calib_data = dict_vid_calib_data[vid_name]
        # ground=1, origin=2, wall=3; sort to index by position
        data = data.sort_values(by="point_type")
        points = data.loc[:, ["x", "y", "type"]].values
        # undistort the annotated points
        undistorted_points = undistort_points(points[:, :2].astype(float), cam_calib_data)
        # store the orig df to keep track of which points are which i.e. ground/wall/origin
        data[["x_undist", "y_undist"]] = undistorted_points.squeeze()
        list_undistorted_points.append(data)
    annotations_df = pd.concat(list_undistorted_points)

    # triangulate all points
    pts_arr = []
    proj_mats = []
    # loop to ensure proj mats and points array match correctly
    for vid_name, cam_calib_data in dict_vid_calib_data.items():
        # get projection matrix from intrinsics and extrinsics
        K = np.array(cam_calib_data["matrix"])
        R = axangle_to_rot_mat(np.array(cam_calib_data["rotation"]))
        # homogenous coordinates
        t = np.array(cam_calib_data["translation"]).reshape(3, 1)
        # since points are undistorted, K = I
        proj_mat = np.hstack((R, t))
        proj_mats.append(proj_mat)
        pts = annotations_df[annotations_df["video"] == vid_name].loc[:, ["x_undist", "y_undist"]].values
        pts_arr.append(pts)

    pts_3d = triangulate_dlt_vectorized(np.stack(pts_arr).squeeze(), np.stack(proj_mats).squeeze())

    # Get the ground and wall planes
    # ground - all points except last (wall)
    z_axis = fit_plane(pts_3d[:-1]) # normalized 
    # wall points z value seem to be smaller than ground points
    y_axis = fit_plane(pts_3d[-2:]) # normalized
    # form orthonormal basis
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    # flip axis so that it matches +ve unit axis convention
    y_axis = - y_axis / np.linalg.norm(y_axis)
    # Cross product to get the 3rd coordinate axis
    x_axis = np.cross(y_axis, z_axis)
    # form rotation matrix; each col is an axis of the global coordinate frame
    R = np.stack((x_axis, y_axis, z_axis), axis=1)
    t = pts_3d[-2] # 2nd last point is the origin
    T = np.hstack((R, t.reshape(3, 1)))
    
    return T