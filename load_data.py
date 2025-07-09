"""
Example script showing how to load and use saved annotations
"""
import numpy as np
import sleap_io as sio
import matplotlib.pyplot as plt

def load_data(labels_file):
    labels = sio.load_slp(labels_file)
    session = labels[session]


def load_annotations(file_path):
    """
    Load annotations from a JSON file
    
    Args:
        file_path (str): Path to the JSON annotation file
        
    Returns:
        dict: Dictionary containing metadata and keypoints
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def convert_to_numpy_format(annotations):
    """
    Convert annotations to numpy array format
    
    Args:
        annotations (dict): Loaded annotations
        
    Returns:
        dict: Dictionary with frame_idx as keys and numpy arrays as values
              Format: {frame_idx: {camera_idx: np.array(shape=(num_keypoints, 2))}}
    """
    keypoints = annotations['keypoints']
    converted = {}
    
    for frame_idx_str, cameras in keypoints.items():
        frame_idx = int(frame_idx_str)
        converted[frame_idx] = {}
        
        for camera_idx_str, keypoints_list in cameras.items():
            camera_idx = int(camera_idx_str)
            if keypoints_list:
                converted[frame_idx][camera_idx] = np.array(keypoints_list, dtype=np.float32)
            else:
                converted[frame_idx][camera_idx] = np.empty((0, 2), dtype=np.float32)
    
    return converted


def get_keypoints_for_frame(annotations, frame_idx, camera_idx=None):
    """
    Get keypoints for a specific frame and optionally a specific camera
    
    Args:
        annotations (dict): Converted annotations
        frame_idx (int): Frame index
        camera_idx (int, optional): Camera index. If None, returns all cameras
        
    Returns:
        np.array or dict: Keypoints for the specified frame/camera
    """
    if frame_idx not in annotations:
        return np.empty((0, 2), dtype=np.float32) if camera_idx is not None else {}
    
    if camera_idx is not None:
        return annotations[frame_idx].get(camera_idx, np.empty((0, 2), dtype=np.float32))
    else:
        return annotations[frame_idx]


def print_annotation_summary(file_path):
    """
    Print a summary of the annotations
    
    Args:
        file_path (str): Path to the annotation file
    """
    data = load_annotations(file_path)
    metadata = data['metadata']
    keypoints = data['keypoints']
    
    print("=== Annotation Summary ===")
    print(f"Number of cameras: {metadata['num_cameras']}")
    print(f"Total frames: {metadata['total_frames']}")
    print(f"FPS: {metadata['fps']}")
    print(f"Video paths: {metadata['video_paths']}")
    
    # Count annotations
    total_annotations = 0
    frames_with_annotations = 0
    
    for frame_idx_str, cameras in keypoints.items():
        frame_has_annotations = False
        for camera_idx_str, keypoints_list in cameras.items():
            if keypoints_list:
                total_annotations += len(keypoints_list)
                frame_has_annotations = True
        if frame_has_annotations:
            frames_with_annotations += 1
    
    print(f"Frames with annotations: {frames_with_annotations}")
    print(f"Total keypoints: {total_annotations}")
    
    # Show some examples
    print("\n=== Example Annotations ===")
    for i, (frame_idx_str, cameras) in enumerate(keypoints.items()):
        if i >= 3:  # Show only first 3 frames
            break
        print(f"Frame {frame_idx_str}:")
        for camera_idx_str, keypoints_list in cameras.items():
            if keypoints_list:
                print(f"  Camera {camera_idx_str}: {len(keypoints_list)} keypoints")
                print(f"    First keypoint: {keypoints_list[0]}")


def main():
    """Example usage"""
    # Example file path (replace with your actual file)
    file_path = "annotations.json"
    
    try:
        # Load and print summary
        print_annotation_summary(file_path)
        
        # Load and convert to numpy format
        data = load_annotations(file_path)
        numpy_annotations = convert_to_numpy_format(data)
        
        # Example: Get keypoints for frame 0, camera 0
        if 0 in numpy_annotations and 0 in numpy_annotations[0]:
            keypoints = numpy_annotations[0][0]
            print(f"\nFrame 0, Camera 0 keypoints shape: {keypoints.shape}")
            print(f"Keypoints: {keypoints}")
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Please run the GUI first to create annotations.")
    except Exception as e:
        print(f"Error loading annotations: {e}")


if __name__ == "__main__":
    main() 