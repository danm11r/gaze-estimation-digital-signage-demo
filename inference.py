import cv2
import logging
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

# switching to mediapipe for face detection
import mediapipe as mp

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze esimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    # face detection provided by the mediapipe open-source project
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            # detect face with mediapipe face detector
            results = face_detector.process(frame)

            # get frame height and width for calculating pixel bbox coordinates from relative bbox
            frame_height, frame_width, _ = frame.shape

            if results.detections:
                for detection in results.detections:
                    rel_bbox = detection.location_data.relative_bounding_box

                    # determine bbox pixel min and max from relative bbox
                    x_min = int(rel_bbox.xmin*frame_width)
                    y_min = int(rel_bbox.ymin*frame_height)

                    x_max = int((rel_bbox.xmin+rel_bbox.width)*frame_width)
                    y_max = int((rel_bbox.ymin+rel_bbox.height)*frame_height)

                    bbox_array = np.array([x_min, y_min, x_max, y_max])

                    image = frame[y_min:y_max, x_min:x_max]
                    image = pre_process(image)
                    image = image.to(device)

                    pitch, yaw = gaze_detector(image)

                    pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                    # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                    # Degrees to Radians
                    pitch_predicted = np.radians(pitch_predicted.cpu())
                    yaw_predicted = np.radians(yaw_predicted.cpu())

                    # draw box and gaze direction
                    draw_bbox_gaze(frame, bbox_array, pitch_predicted, yaw_predicted)

                if params.output:
                    out.write(frame)

                if params.view:
                    cv2.imshow('Demo', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --ouput must be provided.")

    main(args)
