import time
import sys
import os
import json
from pathlib import Path

import torch
import pyrebase
from ultralytics.utils.plotting import Annotator

# # Install YOLOv5 by cloning the repository
# !git clone https://github.com/ultralytics/yolov5

sys.path.append('src/yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    strip_optimizer
)
from yolov5.utils.torch_utils import select_device

# Configuration for Firebase
with open('firebaseConfig.json', 'r') as conf:
    config = json.load(conf)

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

def save_and_upload_video(video_frames, fps, filename, storage):
    local_video_path = f"{filename}.mp4"
    h, w, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create the video writer
    video_writer = cv2.VideoWriter(local_video_path, fourcc, fps, (w, h))

    # Write each frame to the video
    for frame in video_frames:
        video_writer.write(frame)

    video_writer.release()

    # Upload the video to Firebase Storage
    storage.child(f"videos/{filename}.mp4").put(local_video_path)

    # Optionally, remove the local video file after upload
    os.remove(local_video_path)

# Save the image as a JPG and upload it to Firebase Storage
def save_and_upload_frame(image, filename, storage):
    # Save the frame as a .jpg file
    local_image_path = f"{filename}.jpg"
    cv2.imwrite(local_image_path, image)

    # Upload the image to Firebase Storage
    storage.child(f"images/{filename}.jpg").put(local_image_path)

    # Optionally, remove the local image file after upload
    os.remove(local_image_path)

# Modified function run() from yolov5/detect.py
def run_detection(
    weights="model/yolov5s_best.pt",  # model path or triton URL
    source="0",  # file/dir/URL/glob/screen/0(webcam)
    data="yolov5/data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.3,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project="results",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Run YOLOv5 inference for detecting both fire and human objects using two different models (fire and human detection).
    """

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load fire detection model (model for fire detection)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load human detection model (second model)
    model1 = DetectMultiBackend('model/yolov5s.pt', device=device, dnn=dnn, data=data, fp16=half)
    names1 = model1.names

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # Batch size based on number of streams
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Initialize a single vid_writer for the entire session
    vid_writer = None

    # Initialize variables for dynamic FPS calculation
    save_interval = 5
    last_save_time = time.time()

    # Variables to capture and store frames for the 10-second video
    video_frames = []
    video_interval = 10  # Capture and upload a video every 10 seconds
    last_video_time = time.time()

    # Run inference for both models (fire and human detection)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup for fire model
    model1.warmup(imgsz=(1 if pt or model1.triton else bs, 3, *imgsz))  # warmup for human model
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # Set the save_path for the single video output
    save_path = str(save_dir / "output.mp4")

    for i, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # single input for both models
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference for both models (fire and human)
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred_fire = model(im, augment=augment, visualize=visualize)
            pred_human = model1(im, augment=augment, visualize=visualize)

        # Non-Max Suppression (NMS) for both models
        with dt[2]:
            pred_fire = non_max_suppression(pred_fire, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred_human = non_max_suppression(pred_human, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process and annotate the frames
        im0 = im0s[0].copy() if webcam else im0s.copy()
        annotator = Annotator(im0, line_width=line_thickness)

        # Process fire detection results
        for det in pred_fire:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=(0, 0, 255))

        # Process human detection results (only for 'person')
        for det in pred_human:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if names1[int(cls)] == 'person':
                        label = f"{names1[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=(0, 255, 0))

        # Display the results
        im0 = annotator.result()
        if view_img:
            cv2.imshow(str(path), im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save video or image results
        if save_img:
            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_writer is None:
                    w, h = im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
                vid_writer.write(im0)

        # Add the current frame to the video frames list
        video_frames.append(im0)

        # Check if 10 seconds have passed to create and upload a 10-second video
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_and_upload_frame(im0, "camera_1", storage)
            last_save_time = current_time
        if current_time - last_video_time >= video_interval:
            save_and_upload_video(video_frames, 3, "camera_1", storage)
            video_frames = []
            last_video_time = current_time

        seen += 1

    # Release the video writer if used
    if vid_writer:
        vid_writer.release()

    # Print results
    if seen > 0:
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

if __name__ == "__main__":
    # Run YOLOv5 inference to detect fire and people
    run_detection()