import sys
import cv2
import os
import csv
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, 
                             QWidget, QListWidget, QHBoxLayout, QButtonGroup, QGridLayout, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import utils.utils as utils
from utils.keypoint_renderer import KeypointRenderer

class VideoPanel(QWidget):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_clicks = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Video title
        title = QLabel(os.path.basename(self.video_path))
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Video display - will be sized dynamically
        self.video_label = QLabel()
        self.video_label.setStyleSheet("border: 2px solid gray;")
        layout.addWidget(self.video_label)
        
        # Frame counter (read-only)
        self.frame_label = QLabel("Frame: 1 / 1")
        self.frame_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_label)
        
        self.setLayout(layout)
        
        # Connect click event
        self.video_label.mousePressEvent = self.mouse_click
    
    def show_frame(self, frame_number):
        if self.cap is not None and frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Create a copy of the frame to draw on
                display_frame = frame.copy()
                
                # Draw existing clicks for this frame
                if frame_number in self.frame_clicks:
                    for click in self.frame_clicks[frame_number]:
                        if click['point_type'] == 1:
                            # Yellow circle for Ground
                            cv2.circle(display_frame, (click['x'], click['y']), 8, (0, 255, 255), -1)
                        elif click['point_type'] == 2:
                            # Red circle for Origin
                            cv2.circle(display_frame, (click['x'], click['y']), 8, (0, 0, 255), -1)
                        else:
                            # Green circle for Wall
                            cv2.circle(display_frame, (click['x'], click['y']), 8, (0, 255, 0), -1)
                
                # Store the original dimensions for coordinate mapping
                self.original_h, self.original_w = display_frame.shape[:2]
                
                # Set the label size to match the video dimensions
                self.video_label.setFixedSize(self.original_w, self.original_h)
                
                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * self.original_w
                qt_image = QImage(rgb.data, self.original_w, self.original_h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                self.update_frame_display(frame_number)
    
    def update_frame_display(self, frame_number):
        self.frame_label.setText(f"Frame: {frame_number + 1} / {self.total_frames}")
    
    def mouse_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        
        # Get the parent's current point type and frame number
        parent = self.parent()
        while parent and not hasattr(parent, 'current_point_type'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'current_frame_number'):
            point_type = parent.current_point_type
            frame_number = parent.current_frame_number
            print(f"Clicked at: ({x}, {y}) on {os.path.basename(self.video_path)} frame {frame_number + 1} with point type {point_type}")
            
            # Check if click is within the video area
            if 0 <= x < self.original_w and 0 <= y < self.original_h:
                print(f"Click coordinates: ({x}, {y}) - using directly as original video coordinates")
                
                # Store click for this specific frame
                if frame_number not in self.frame_clicks:
                    self.frame_clicks[frame_number] = []
                
                self.frame_clicks[frame_number].append({
                    'x': x, 
                    'y': y, 
                    'point_type': point_type
                })
                
                # Add to parent's overall clicks list
                parent.clicks.append({
                    'video': os.path.basename(self.video_path),
                    'frame': frame_number + 1,
                    'x': x,
                    'y': y,
                    'point_type': point_type
                })
                
                # Redraw frame to show the new click
                self.show_frame(frame_number)
            else:
                print(f"Click outside video area: ({x}, {y})")
    
    def get_all_clicks(self):
        """Return all clicks for this video"""
        all_clicks = []
        for frame_num, clicks in self.frame_clicks.items():
            for click in clicks:
                all_clicks.append({
                    'video': os.path.basename(self.video_path),
                    'frame': frame_num + 1,
                    'x': click['x'],
                    'y': click['y'],
                    'point_type': click['point_type']
                })
        return all_clicks

class VideoClickApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Video Click Tracker")
        self.setMinimumSize(800, 600)

        self.layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        self.loadButton = QPushButton("Load MP4 Videos")
        self.loadCalibButton = QPushButton("Load Calibration")
        self.loadTargetsButton = QPushButton("Load Targets")
        self.loadClicksButton = QPushButton("Load Clicks")
        self.saveButton = QPushButton("Save Clicks")
        self.saveVideoButton = QPushButton("Save Video")
        self.registerButton = QPushButton("Register")
        self.renderButton = QPushButton("Render")
        
        # Point type selection
        self.pointTypeLabel = QLabel("Point Type:")
        self.groundButton = QPushButton("Ground")
        self.originButton = QPushButton("Origin")
        self.wallButton = QPushButton("Wall")
        self.groundButton.setCheckable(True)
        self.originButton.setCheckable(True)
        self.wallButton.setCheckable(True)
        self.groundButton.setChecked(True)  # Default to Ground
        
        control_layout.addWidget(self.loadButton)
        control_layout.addWidget(self.loadCalibButton)
        control_layout.addWidget(self.loadTargetsButton)
        control_layout.addWidget(self.loadClicksButton)
        control_layout.addWidget(self.pointTypeLabel)
        control_layout.addWidget(self.groundButton)
        control_layout.addWidget(self.originButton)
        control_layout.addWidget(self.wallButton)
        control_layout.addWidget(self.saveButton)
        control_layout.addWidget(self.saveVideoButton)
        control_layout.addWidget(self.registerButton)
        control_layout.addWidget(self.renderButton)
        control_layout.addStretch()
        
        self.layout.addLayout(control_layout)
        
        # Global frame navigation
        frame_nav_layout = QHBoxLayout()
        self.prevButton = QPushButton("Previous Frame")
        self.nextButton = QPushButton("Next Frame")
        self.frameLabel = QLabel("Frame: 0 / 0")
        
        frame_nav_layout.addWidget(self.prevButton)
        frame_nav_layout.addWidget(self.frameLabel)
        frame_nav_layout.addWidget(self.nextButton)
        frame_nav_layout.addStretch()
        
        self.layout.addLayout(frame_nav_layout)
        
        # Video panels area
        self.scroll_area = QScrollArea()
        self.video_widget = QWidget()
        self.video_layout = QGridLayout()
        self.video_widget.setLayout(self.video_layout)
        self.scroll_area.setWidget(self.video_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)
        
        self.setLayout(self.layout)

        self.loadButton.clicked.connect(self.load_videos)
        self.loadCalibButton.clicked.connect(self.load_calibration)
        self.loadTargetsButton.clicked.connect(self.load_targets)
        self.loadClicksButton.clicked.connect(self.load_clicks)
        self.saveButton.clicked.connect(self.save_clicks)
        self.saveVideoButton.clicked.connect(self.set_video_save_path)
        self.registerButton.clicked.connect(self.register)
        self.renderButton.clicked.connect(self.render)
        self.prevButton.clicked.connect(self.previous_frame)
        self.nextButton.clicked.connect(self.next_frame)
        self.groundButton.clicked.connect(lambda: self.set_point_type(1))
        self.originButton.clicked.connect(lambda: self.set_point_type(2))
        self.wallButton.clicked.connect(lambda: self.set_point_type(3))

        self.video_files = []
        self.clicks = []
        self.video_panels = []
        self.current_point_type = 1  # Default point type (Ground)
        self.current_frame_number = 0
        self.max_frames = 0
        
        # File paths and loaded data
        self.video_save_path = None
        self.calibration_file = None
        self.targets = None
        self.edges = [
            (0,2), # nose to left ear
            (0,1), # nose to right ear
            (2,5), # left ear to head
            (1,5), # right ear to head
            (5,12), # head to haunch right
            (5,13), # head to haunch left
            (12,3), # haunch right to TTI
            (13,3), # haunch left to TTI
            (6,3), # trunk to TTI
            (3,7), # TTI to tail 0
            (7,8), # tail 0 to tail 1
            (8,9), # tail 1 to tail 2
        ]
        
        # Store registration results
        self.targets_global = None
        self.T = None
        
        # Store clicks DataFrame
        self.clicks_df = None
        self.clicks_user_input = False
        
        # Set initial button states
        self.update_button_states()

    def update_button_states(self):
        """Update button enabled/disabled states based on internal state."""
        # Register button requires: calibration, targets, and clicks
        can_register = (self.calibration_file is not None and 
                       self.targets is not None and 
                       self.clicks_df is not None)
        self.registerButton.setEnabled(can_register)
        
        # Render button requires: registration data and video save path
        can_render = (self.targets_global is not None and 
                     self.T is not None and 
                     self.video_save_path is not None)
        self.renderButton.setEnabled(can_render)
        
        # Save clicks button requires: video panels with clicks
        has_clicks = any(panel.frame_clicks for panel in self.video_panels)
        self.saveButton.setEnabled(has_clicks)
        
        # Navigation buttons require: loaded videos
        has_videos = len(self.video_panels) > 0
        self.prevButton.setEnabled(has_videos and self.current_frame_number > 0)
        self.nextButton.setEnabled(has_videos and self.current_frame_number < self.max_frames - 1)

    def set_point_type(self, point_type):
        self.current_point_type = point_type
        # Update button states
        self.groundButton.setChecked(point_type == 1)
        self.originButton.setChecked(point_type == 2)
        self.wallButton.setChecked(point_type == 3)
        self.update_button_states() # Call the new method

    def load_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select MP4 files", "", "Video Files (*.mp4)")
        if files:
            self.video_files = files
            self.create_video_panels()

    def load_calibration(self):
        """Load calibration file and store in memory."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration File", "", "TOML Files (*.toml);;All Files (*)")
        if file_path:
            try:
                self.calibration_file = utils.load_calibration_data(file_path)
                print(f"Calibration file loaded successfully from: {file_path}")
            except Exception as e:
                print(f"Error loading calibration file: {str(e)}")
                self.calibration_file = None
        self.update_button_states()

    def load_targets(self):
        """Load targets file and store as numpy array."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Targets File", "", "NumPy Files (*.npy);;All Files (*)")
        if file_path:
            try:
                self.targets = np.load(file_path)
                print(f"Targets file loaded successfully from: {file_path}")
                print(f"Targets shape: {self.targets.shape}")
            except Exception as e:
                print(f"Error loading targets file: {str(e)}")
                self.targets = None
        self.update_button_states()

    def load_clicks(self):
        """Load clicks file and store as DataFrame."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Clicks File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                self.clicks_df = pd.read_csv(file_path)
                print(f"Clicks file loaded successfully from: {file_path}")
                self.clicks_user_input = True
                print(f"Loaded {len(self.clicks_df)} clicks")
                print(f"Columns: {list(self.clicks_df.columns)}")
            except Exception as e:
                print(f"Error loading clicks file: {str(e)}")
                self.clicks_df = None
        self.update_button_states()

    def set_video_save_path(self):
        """Set the path where the rendered video will be saved."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "MP4 Files (*.mp4)")
        if file_path:
            self.video_save_path = file_path
            print(f"Video will be saved to: {self.video_save_path}")
        else:
            print("No save path selected. Video will use default path.")
        self.update_button_states()

    def create_video_panels(self):
        # Clear existing panels
        for panel in self.video_panels:
            panel.deleteLater()
        self.video_panels.clear()
        
        # Clear layout
        while self.video_layout.count():
            child = self.video_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create new panels
        cols = 3  # Number of columns in the grid
        for i, video_path in enumerate(self.video_files):
            panel = VideoPanel(video_path, self)
            row = i // cols
            col = i % cols
            self.video_layout.addWidget(panel, row, col)
            self.video_panels.append(panel)
        
        # Find the minimum number of frames across all videos
        if self.video_panels:
            self.max_frames = min(panel.total_frames for panel in self.video_panels)
            self.current_frame_number = 0
            self.update_frame_display()
            self.show_all_frames()
            self.update_button_states() 

    def show_all_frames(self):
        """Show the current frame on all panels"""
        for panel in self.video_panels:
            panel.show_frame(self.current_frame_number)

    def update_frame_display(self):
        self.frameLabel.setText(f"Frame: {self.current_frame_number + 1} / {self.max_frames}")

    def previous_frame(self):
        if self.current_frame_number > 0:
            self.current_frame_number -= 1
            self.update_frame_display()
            self.show_all_frames()
            self.update_button_states() 

    def next_frame(self):
        if self.current_frame_number < self.max_frames - 1:
            self.current_frame_number += 1
            self.update_frame_display()
            self.show_all_frames()
            self.update_button_states() 

    def save_clicks(self):
        # Collect all clicks from all panels
        all_clicks = []
        for panel in self.video_panels:
            all_clicks.extend(panel.get_all_clicks())
        
        # Convert to DataFrame and store in internal state
        self.clicks_df = pd.DataFrame(all_clicks)
        
        # Save to CSV file
        path, _ = QFileDialog.getSaveFileName(self, "Save Clicks", "", "CSV Files (*.csv)")
        if path:
            clicks_df_processed = utils.process_raw_annotations(self.clicks_df)
            clicks_df_processed.to_csv(path, index=False)
            print(f"Saved {len(all_clicks)} clicks to {path}.")
        
        self.update_button_states() 

    def register(self):
        """Execute the registration pipeline."""
        try:
            print("Starting registration process...")
            
            # 1. Use loaded targets
            print("Using loaded targets...")
            print(f"Targets shape: {self.targets.shape}")
            
            # 2. Compute global coordinate frame
            print("Computing global coordinate frame...")
            T = utils.compute_global_coordinate_frame(
                calib_data=self.calibration_file,
                annotations=self.clicks_df
            )
            print("Global coordinate frame computed successfully")
            
            # 3. Register targets to global frame
            print("Registering targets to global frame...")
            targets_global = utils.register_to_global_frame(self.targets, T)
            print(f"Targets registered with shape: {targets_global.shape}")
            
            # 4. Store the results for later rendering
            self.targets_global = targets_global
            self.T = T
            
            print("Registration completed successfully! Click 'Render' to create the 3D video.")
            
        except Exception as e:
            print(f"Error during registration: {str(e)}")
            import traceback
            traceback.print_exc()
        self.update_button_states()


    def render(self, fps=30):
        """Execute the 3D rendering pipeline."""
        try:
            # 1. Define edges for the keypoint renderer (you may need to adjust this)
            # This is a simple example - you might need to define the actual edges for your keypoints
            edges = []  # Add your edge definitions here if needed
            
            # 2. Initialize renderer and create video
            print("Initializing renderer...")
            renderer = KeypointRenderer(self.targets_global, self.edges, self.T)
            renderer.set_style(node_size=50, edge_width=4, alpha=0.9)
            
            # 3. Start recording
            print("Starting video recording...")
            renderer.start_recording(fps=fps)
            
            # 4. Create plot and render frames
            print("Creating 3D plot and rendering frames...")
            fig, ax = renderer.create_3d_plot(
                figsize=(12, 10),
                title="3D Mouse Pose - Frame 0",
                view_elev=45,
                view_azim=45
            )
            
            # 5. Render all frames
            total_frames = self.targets_global.shape[0]
            for frame_idx in range(total_frames):
                if frame_idx % 100 == 0:  # Progress indicator
                    print(f"Rendering frame {frame_idx}/{total_frames}")
                renderer.update_3d_plot(frame_idx)
                renderer.capture_frame()
            
            # 6. Stop recording and save
            print("Stopping recording and saving video...")
            renderer.stop_recording()
            renderer.save_video(self.video_save_path)
            
            print("Rendering completed successfully!")
            
        except Exception as e:
            print(f"Error during rendering: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoClickApp()
    window.show()
    sys.exit(app.exec_())
