from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2

model = YOLO("best.pt")
names = model.model.names

cap = cv2.VideoCapture("1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

line_pts = [(0, 360), (1280, 360)]
ind=0
# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=False, show=False, conf=0.1, tracker="bytetrack.yaml")

    im0 = speed_obj.estimate_speed(im0, tracks)
    
    cv2.imwrite(f'./frames2/{ind}.jpg', im0)
    ind+=1
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
