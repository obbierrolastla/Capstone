from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('E:\ITS\Capstone\Codes\YOLO-based\license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')
# cap = cv2.VideoCapture(0)

vehicles = [2, 3, 5, 7]

# read frames   
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print(license_plate_text if license_plate_text else '-')
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, 'test.csv')

# import threading
# import numpy as np
# import cv2
# from ultralytics import YOLO

# import util
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv

# import time

# results = {}
# stop_flag = False

# def listen_for_quit():
#     global stop_flag
#     input("Press Enter to stop capture...\n")
#     stop_flag = True

# # Start listener in background
# threading.Thread(target=listen_for_quit, daemon=True).start()

# mot_tracker = Sort()

# # load models
# coco_model = YOLO('yolov8n.pt')
# license_plate_detector = YOLO(r'E:\ITS\Capstone\Codes\YOLO-based\license_plate_detector.pt')

# # capture from webcam (0 = default camera)
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# vehicles = [2, 3, 5, 7]

# frame_nmr = -1

# time_prev = time.time()                
# while True:
#     ret, frame = cap.read()
#     if not ret or stop_flag:
#         break

#     frame_nmr += 1
#     results[frame_nmr] = {}

#     # detect vehicles
#     detections = coco_model(frame)[0]
#     detections_ = []
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])

#     # track vehicles
#     if len(detections_) > 0:
#         track_ids = mot_tracker.update(np.asarray(detections_))
#     else:
#         track_ids = mot_tracker.update(np.empty((0, 5)))

#     # detect license plates
#     license_plates = license_plate_detector(frame)[0]
#     license_plate_text, license_plate_text_score = None, None
    
#     for license_plate in license_plates.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate

#         # assign license plate to car
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#         if car_id != -1:
#             # crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

#             # process license plate
#             license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#             _, license_plate_crop_thresh = cv2.threshold(
#                 license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
#             )

#             # read license plate number
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

#             if license_plate_text is not None:
#                 results[frame_nmr][car_id] = {
#                     'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                     'license_plate': {
#                         'bbox': [x1, y1, x2, y2],
#                         'text': license_plate_text,
#                         'bbox_score': score,
#                         'text_score': license_plate_text_score
#                     }
#                 }

#         print(f"Frame {frame_nmr} | Car ID: {car_id} | Plate: {license_plate_text if license_plate_text else '-'}")

# cap.release()
# write_csv(results, 'test.csv')
# print("Done. Results saved to test.csv")