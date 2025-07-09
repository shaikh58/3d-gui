import sys
import cv2
import os
import csv
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, 
                             QWidget, QListWidget, QHBoxLayout, QButtonGroup, QGridLayout, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import numpy as np

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
        self.saveButton = QPushButton("Save Clicks")
        
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
        control_layout.addWidget(self.pointTypeLabel)
        control_layout.addWidget(self.groundButton)
        control_layout.addWidget(self.originButton)
        control_layout.addWidget(self.wallButton)
        control_layout.addWidget(self.saveButton)
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
        self.saveButton.clicked.connect(self.save_clicks)
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

    def set_point_type(self, point_type):
        self.current_point_type = point_type
        # Update button states
        self.groundButton.setChecked(point_type == 1)
        self.originButton.setChecked(point_type == 2)
        self.wallButton.setChecked(point_type == 3)

    def load_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select MP4 files", "", "Video Files (*.mp4)")
        if files:
            self.video_files = files
            self.create_video_panels()

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

    def next_frame(self):
        if self.current_frame_number < self.max_frames - 1:
            self.current_frame_number += 1
            self.update_frame_display()
            self.show_all_frames()

    def save_clicks(self):
        # Collect all clicks from all panels
        all_clicks = []
        for panel in self.video_panels:
            all_clicks.extend(panel.get_all_clicks())
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Clicks", "", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['video', 'frame', 'x', 'y', 'point_type'])
                writer.writeheader()
                writer.writerows(all_clicks)
            print(f"Saved {len(all_clicks)} clicks to {path}.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoClickApp()
    window.show()
    sys.exit(app.exec_())
