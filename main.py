import sys
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO

source_img_path = sys.argv[1]
yolo_device = "cpu"
out_file_name = "result.jpg" if sys.argv[2] is None else sys.argv[2]

print(out_file_name)

### Detect

print("Starting primary container detection...")

detect_model = YOLO("detect_model.pt")
detect_results = \
    detect_model.predict(source=source_img_path, device=yolo_device, conf=0.4, iou=0.5, imgsz=1280, save=True,
                         project="./", name="intermediate_detect", exist_ok=True)[0]
print()

if len(detect_results.boxes) == 0:
    print("No containers were detected. Exiting...")
    sys.exit(1)

print("Finding closest container...")

img_dim = np.array(detect_results.orig_shape)[::-1]
closest_to = np.array([img_dim[0] / 2, 0])

boxes = np.array(detect_results.boxes.xyxy)

points = [np.split(box, 2) for box in boxes]
distances = list(map(lambda x: min(list(map(lambda y: np.linalg.norm(y), list(x)))), list(closest_to - points)))
closest_box_idx = np.argmin(distances)
closest_box_xyxy = boxes[closest_box_idx]

### Crop

print("Cropping image...")

img = Image.open(source_img_path)
offset = 32
crop_coords = (lambda x: [x[0] - offset, 0, x[2] + offset, x[3] + offset])(closest_box_xyxy)
cropped_img = img.crop(crop_coords)
crop_save_path = "intermediate_crop/tmp.png"

os.makedirs("intermediate_crop", exist_ok=True)
cropped_img.save(crop_save_path)

### Pose

print("Starting keypoint detection...")

pose_model = YOLO("pose_model.pt")
pose_results = pose_model.predict(source=crop_save_path, device=yolo_device, conf=0.3, iou=0., imgsz=256, save=True,
                                  project="./", name="intermediate_pose", exist_ok=True, show_labels=False,
                                  show_boxes=False)[0]
print()
if len(pose_results.boxes) == 0:
    print("Keypoint detection failed. Make sure hook is present in the original photo. Exiting...")
    sys.exit(1)

keypoints = pose_results.keypoints

print(f"Found keypoints with confidence scores: {keypoints.conf}")

### Angles

print("Starting angle calculation...")

keypoints_global_xy = \
    list(map(lambda x: list(np.array(x) + np.array([crop_coords[0], crop_coords[1]])), keypoints.xy))[0]

box_pts_cr = keypoints_global_xy[:-1]
hook_pt_cr = keypoints_global_xy[-1]
box_pts_rc = []

for box_pt_cr in box_pts_cr:
    box_pts_rc.append(box_pt_cr[::-1])
hook_pt_rc = hook_pt_cr[::-1]

fov = 0.2
a, b, c = [2.5908, 3.048, 2.4384]


def arr_to_str(arr):
    return "{" + str(arr[0]) + ", " + str(arr[0]) + "}"


script_hook_str = "\"" + arr_to_str(hook_pt_rc) + "\""

script_box_str = "\"{"
for box_pt_rc in box_pts_rc:
    script_box_str += arr_to_str(box_pt_rc) + ", "
script_box_str = script_box_str[:len(script_box_str)-2]
script_box_str += "}\""

wolfram_command = f"./script {source_img_path} {fov} {a} {b} {c} {script_box_str} {script_hook_str} {out_file_name} True"

os.system(wolfram_command)

print(f"Done. Check {out_file_name}")
