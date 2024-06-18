import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PySide6.QtGui import QPixmap
from PIL import Image
import numpy as np
import torch 
from predict import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_unet()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
# Placeholder function for your deep learning model processing
def process_image(input_image_path):
    # Perform your model's processing here
    # Load the image and process
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    x = np.transpose(image, (2, 0, 1))
    x = x/255.0
    x = np.expand_dims(x, axis = 0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    
    with torch.no_grad():
        # Predict
        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred[0].cpu().numpy() 
        y_pred = np.squeeze(y_pred, axis = 0)
        y_pred = y_pred > 0.5 
        y_pred = np.array(y_pred, dtype=np.uint8)

    y_pred = mask_parse(y_pred)
    y_pred = y_pred*255
    
    # Save the putput image to a temporary file
    output_image_path = 'processed_image.png'
    cv2.imwrite(output_image_path, y_pred)
    
    return output_image_path

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the main layout
        main_layout = QVBoxLayout()

        # Set up the image layout to hold the input and output images side by side
        image_layout = QHBoxLayout()
        
        # Input image label
        self.inputImageLabel = QLabel("No image selected")
        self.inputImageLabel.setFixedSize(400, 400)
        self.inputImageLabel.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.inputImageLabel)

        # Output image label
        self.outputImageLabel = QLabel("Output will be displayed here")
        self.outputImageLabel.setFixedSize(400, 400)
        self.outputImageLabel.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.outputImageLabel)

        # Add the image layout to the main layout
        main_layout.addLayout(image_layout)

        # Select button
        self.selectButton = QPushButton("Select Image")
        self.selectButton.clicked.connect(self.openFileNameDialog)
        main_layout.addWidget(self.selectButton)

        # Process button
        self.processButton = QPushButton("Process")
        self.processButton.clicked.connect(self.processImage)
        main_layout.addWidget(self.processButton)

        self.setLayout(main_layout)
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 850, 500)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            self.inputImagePath = fileName
            self.inputImageLabel.setPixmap(QPixmap(fileName).scaled(400, 400))

    def processImage(self):
        if hasattr(self, 'inputImagePath'):
            outputImagePath = process_image(self.inputImagePath)
            self.outputImageLabel.setPixmap(QPixmap(outputImagePath).scaled(400, 400))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessorApp()
    ex.show()
    sys.exit(app.exec())
