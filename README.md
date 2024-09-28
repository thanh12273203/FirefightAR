# FirefightAR 🔥🧑‍🚒

FirefightAR is an Augmented Reality (AR) application designed to assist firefighters in detecting and responding to fire hazards in real-time using YOLOv5 object detection. The application utilizes computer vision to detect fire and people, providing valuable insights for rescue and fire containment efforts.

## Features
- **Real-time Object Detection**: Detects fire and human objects in real-time from a live camera feed.
- **YOLOv5 Integration**: Leverages YOLOv5 for efficient and accurate object detection.
- **Firebase Integration**: Automatically saves detected images (every 5 seconds) to Firebase Storage for further analysis or archiving.
- **Augmented Reality Interface**: Helps visualize critical information, such as fire locations, in an AR environment for better situational awareness.

## Getting Started

### Prerequisites
Before running the application, ensure that the following are installed on your system:
- Python 3.12.x
- Git
- A working camera (for real-time video capture)
- Firebase account (for storing images)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/thanh12273203/FirefightAR.git
   cd FirefightAR
   ```
   
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
3. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
   
4. **Configure Firebase**:
   - Update the Firebase configuration in `src/main.py` with your Firebase project credentials.
   - Add your `serviceAccount.json` file to the `src/` directory to enable Firebase integration.

5. **Download YOLOv5 model weights**:
   - Download the pre-trained YOLOv5 model (e.g., `yolov5s.pt`) and place it in the `model/` folder.
   - Alternatively, train a custom YOLOv5 model and place the weights in the model/ folder.

### Running the Application
To start the application and begin detecting fire and people in real-time using the webcam, run:
```bash
python src/main.py
```

The application will:
- Open the web-cam and start real-time object detection.
- Save a snapshot every 5 seconds and upload it to Firebase Storage

### Results
Videos with bounding boxes will be saved locally in the `results/` folder and frames will be uploaded to Firebase Storage under the `images/` path.

## Acknowledgements
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the powerful object detection framework.
- The pre-trained model `yolov5s_best.pt` for fire detection was downloaded from [this repo](https://github.com/spacewalk01/yolov5-fire-detection).
- [Firebase](https://firebase.google.com/) for cloud storage and real-time database services.
