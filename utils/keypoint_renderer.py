import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.cm as cm
import io
from PIL import Image
from IPython.display import clear_output, display

class KeypointRenderer:
    """
    A class to render keypoints connected by edges to form an instance outline.
    Supports both 2D and 3D visualization with customizable styling.
    """
    
    def __init__(self, keypoints, edges, T=None):
        """
        Initialize the renderer with keypoints and edges.
        
        Args:
            keypoints: numpy array of shape (N, K, 3)
            edges: list of tuples (i, j) where i and j are keypoint indices
            T: numpy array of shape (4, 4) containing pose matrix of global frame
        """
        self.keypoints = keypoints
        self.edges = edges
        self.T = T
        
        # Default styling
        # self.node_color = '#FF6B6B'  # Coral red
        # self.edge_color = '#4ECDC4'  # Turquoise
        self.node_size = 100
        self.edge_width = 3
        self.alpha = 0.8
        
        # Coordinate axes styling
        self.axis_length = 50  # Length of coordinate axes
        self.axis_width = 3    # Width of coordinate axes
        
        # Store figure and axes for reuse
        self.fig = None
        self.ax = None
        self.plot_objects = {}  # Store plot objects for updating
        
        # Video recording attributes
        self.frames_buffer = []
        self.recording = False
        self.video_writer = None
        self.fps = 30

    def set_style(self, node_color=None, edge_color=None, node_size=None, 
                  edge_width=None, alpha=None, axis_length=None, axis_width=None):
        """Update the visual style of the renderer."""
        if node_color is not None:
            self.node_color = node_color
        if edge_color is not None:
            self.edge_color = edge_color
        if node_size is not None:
            self.node_size = node_size
        if edge_width is not None:
            self.edge_width = edge_width
        if alpha is not None:
            self.alpha = alpha
        if axis_length is not None:
            self.axis_length = axis_length
        if axis_width is not None:
            self.axis_width = axis_width

    def start_recording(self, fps=30):
        """
        Start recording frames to buffer for video creation.
        
        Args:
            fps: frames per second for the output video
        """
        self.frames_buffer = []
        self.recording = True
        self.fps = fps
        print(f"Started recording at {fps} FPS")

    def stop_recording(self):
        """Stop recording frames."""
        self.recording = False
        print(f"Stopped recording. Captured {len(self.frames_buffer)} frames")

    def capture_frame(self):
        """
        Capture the current figure as a frame and add to buffer if recording.
        """
        if not self.recording or self.fig is None:
            return
        
        # Convert matplotlib figure to image
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=75, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Convert to PIL Image and then to numpy array
        img = Image.open(buf)
        img_array = np.array(img)
        
        # Convert RGBA to BGR (OpenCV format)
        if img_array.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        self.frames_buffer.append(img_bgr)
        buf.close()

    def save_video(self, output_path, fps=None):
        """
        Save the recorded frames as an MP4 video.
        
        Args:
            output_path: path where to save the video file
            fps: frames per second (uses recording fps if not specified)
        """
        if not self.frames_buffer:
            print("No frames recorded. Call start_recording() and capture frames first.")
            return
        
        if fps is None:
            fps = self.fps
        
        # Get frame dimensions from the first frame
        height, width = self.frames_buffer[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for i, frame in enumerate(self.frames_buffer):
            self.video_writer.write(frame)
            # if (i + 1) % 10 == 0:  # Progress indicator
            #     print(f"Writing frame {i + 1}/{len(self.frames_buffer)}")
        
        # Release video writer
        self.video_writer.release()
        print(f"Video saved to: {output_path}")
        print(f"Video details: {len(self.frames_buffer)} frames, {fps} FPS, {width}x{height}")

    def create_animation(self, output_path, fps=30, show_labels=False, 
                        show_coordinate_axes=True, show_progress=True):
        """
        Create and save an animation of all frames automatically.
        
        Args:
            output_path: path where to save the video file
            fps: frames per second for the output video
            show_labels: whether to show keypoint labels
            show_coordinate_axes: whether to show coordinate axes
            show_progress: whether to show progress during rendering
        """
        # Start recording
        self.start_recording(fps)
        
        # Create the initial plot
        self.create_3d_plot()
        
        # Get total number of frames
        total_frames = self.keypoints.shape[0]
        
        # Render each frame
        for frame_idx in range(total_frames):
            
            # Update the plot
            self.update_3d_plot(frame_idx, show_labels, show_coordinate_axes)
            
            # Capture the frame
            self.capture_frame()
        
        # Stop recording and save video
        self.stop_recording()
        self.save_video(output_path, fps)
        
        # Clear buffer to free memory
        self.frames_buffer = []

    # ... existing methods remain the same ...
    def draw_coordinate_axes(self):
        """
        Draw coordinate axes for a given transformation matrix.
        
        Args:
            T: 4x4 transformation matrix
        """
        # Extract rotation and translation
        R = np.eye(3)  # Rotation matrix
        t = np.zeros(3)   # Translation vector
        
        # Define unit vectors in local coordinate system
        x_axis = np.array([self.axis_length, 0, 0])
        y_axis = np.array([0, self.axis_length, 0])
        z_axis = np.array([0, 0, self.axis_length])
        
        # Transform to world coordinates
        x_world = R @ x_axis + t
        y_world = R @ y_axis + t
        z_world = R @ z_axis + t
        
        # Draw axes with different colors
        self.ax.plot([t[0], x_world[0]], [t[1], x_world[1]], [t[2], x_world[2]], 
                    color='red', linewidth=self.axis_width, alpha=self.alpha, 
                    label="x")
        self.ax.plot([t[0], y_world[0]], [t[1], y_world[1]], [t[2], y_world[2]], 
                    color='green', linewidth=self.axis_width, alpha=self.alpha,
                    label="y")
        self.ax.plot([t[0], z_world[0]], [t[1], z_world[1]], [t[2], z_world[2]], 
                    color='blue', linewidth=self.axis_width, alpha=self.alpha,
                    label="z")
        
        # Add small spheres at the origin and tips
        self.ax.scatter([t[0]], [t[1]], [t[2]], color='black', s=50, alpha=self.alpha)
        self.ax.scatter([x_world[0]], [x_world[1]], [x_world[2]], color='red', s=30, alpha=self.alpha)
        self.ax.scatter([y_world[0]], [y_world[1]], [y_world[2]], color='green', s=30, alpha=self.alpha)
        self.ax.scatter([z_world[0]], [z_world[1]], [z_world[2]], color='blue', s=30, alpha=self.alpha)

    
    def create_3d_plot(self, figsize=(10,8), show_labels=False, show_grid=False,
                       title=None, view_elev=20, view_azim=45):
        """
        Create a 3D plot with axes and styling. Call this once before rendering multiple frames.
        
        Args:
            figsize: tuple for figure size
            show_labels: whether to show keypoint labels
            show_grid: whether to show grid
            title: plot title
            view_elev: elevation angle for 3D view
            view_azim: azimuth angle for 3D view
        """
        if self.keypoints.shape[-1] < 3:
            raise ValueError("Keypoints must have at least 3 dimensions for 3D rendering")
        
        # Create figure and axes
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Get the number of mice
        num_instances = self.keypoints.shape[1]
        
        # Define colors for different mice using a color palette
        color_map = cm.tab10
        colors = [color_map(i) for i in range(num_instances)]
        
        # Initialize plot objects storage
        self.plot_objects = {
            'edges': [[] for _ in range(num_instances)],
            'nodes': [[] for _ in range(num_instances)],
            'labels': [[] for _ in range(num_instances)],
            'colors': colors
        }
        
        # Set up styling
        if show_grid:
            self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        
        # Set view angle
        self.ax.view_init(elev=view_elev, azim=view_azim)
        
        # Set plot limits to the minimum and maximum values without padding (ignoring NaN)
        x_min, x_max = min(np.nanmin(self.keypoints[:, :, :, 0]), 0), max(np.nanmax(self.keypoints[:, :, :, 0]), self.T[0,3])
        y_min, y_max = min(np.nanmin(self.keypoints[:, :, :, 1]), 0), max(np.nanmax(self.keypoints[:, :, :, 1]), self.T[1,3])
        z_min, z_max = min(np.nanmin(self.keypoints[:, :, :, 2]), 0), max(np.nanmax(self.keypoints[:, :, :, 2]), self.T[2,3])
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        
        # Add legend if multiple instances
        if num_instances > 1:
            self.ax.legend()
        
        plt.tight_layout()
        return self.fig, self.ax

    
    
    def update_3d_plot(self, frame_idx, show_labels=False, show_coordinate_axes=True):
        """
        Update the 3D plot with data from a specific frame. 
        Call create_3d_plot() first, then call this method in a loop.
        
        Args:
            frame_idx: frame index to render
            show_labels: whether to show keypoint labels
            show_coordinate_axes: whether to show coordinate axes
        """
        if self.fig is None or self.ax is None:
            raise ValueError("Call create_3d_plot() first before updating")
        
        # Clear previous frame data
        self.ax.clear()
        
        # Get the number of mice in this frame
        num_instances = self.keypoints.shape[1]
        
        # Plot each instance
        for inst_idx in range(num_instances):
            instance_color = self.plot_objects['colors'][inst_idx]
            
            # Plot edges (lines connecting keypoints) for this instance
            for edge in self.edges:
                start_idx, end_idx = edge
                start_point = self.keypoints[frame_idx, inst_idx, start_idx]
                end_point = self.keypoints[frame_idx, inst_idx, end_idx]
                
                self.ax.plot([start_point[0], end_point[0]], 
                           [start_point[1], end_point[1]], 
                           [start_point[2], end_point[2]], 
                           color=instance_color, linewidth=self.edge_width, 
                           alpha=self.alpha)
            
            # Plot keypoints (nodes) for this instance
            self.ax.scatter(self.keypoints[frame_idx, inst_idx, :, 0], 
                          self.keypoints[frame_idx, inst_idx, :, 1], 
                          self.keypoints[frame_idx, inst_idx, :, 2],
                          c=instance_color, s=self.node_size, alpha=self.alpha, 
                          edgecolors='white', linewidth=2, label=f'Instance {inst_idx + 1}')
            
            # Add labels if requested
            if show_labels:
                for i, point in enumerate(self.keypoints[frame_idx, inst_idx]):
                    self.ax.text(point[0], point[1], point[2], f'{inst_idx}_{i}', 
                               fontsize=6, fontweight='bold')
            
        # Draw coordinate axes if requested and available
        if show_coordinate_axes and self.T is not None:
            self.draw_coordinate_axes()
        
        # Set plot limits to the minimum and maximum values without padding (ignoring NaN)
        x_min, x_max = min(np.nanmin(self.keypoints[:, :, :, 0]), 0), max(np.nanmax(self.keypoints[:, :, :, 0]), self.T[0,3])
        y_min, y_max = min(np.nanmin(self.keypoints[:, :, :, 1]), 0), max(np.nanmax(self.keypoints[:, :, :, 1]), self.T[1,3])
        z_min, z_max = min(np.nanmin(self.keypoints[:, :, :, 2]), 0), max(np.nanmax(self.keypoints[:, :, :, 2]), self.T[2,3])
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)

        # Reapply styling
        self.ax.set_title(f"Mouse Keypoint Skeleton (3D) - Frame {frame_idx}", 
                         fontsize=14, fontweight='bold', pad=20)
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        
        # Add legend if multiple instances
        if num_instances > 1:
            self.ax.legend()
        
        # Force redraw - multiple methods to ensure update in notebooks
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to allow the plot to update
        
        # Alternative method for notebook compatibility
        # try:
        #     clear_output(wait=True)
        #     display(self.fig)
        # except ImportError:
        #     pass  # Not in a notebook environment