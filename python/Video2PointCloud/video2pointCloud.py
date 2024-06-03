import cv2
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import timm

# function that recieves an image and return the depth representation of that image
def getDepth(img):
    #img_resized = cv2.resize(img, (320, 240))  # Reducing the frame size
    img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_t).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Resize back to original size
    depth_resized = cv2.resize(depth, (img.shape[1], img.shape[0]))

    return depth_resized

# models
#MiDaS_small
# DPT_Large
#DPT_Hybrid
model_type = "MiDaS_small"

# loading the model 
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# setting the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# loading the input transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


cap = cv2.VideoCapture("cat.mp4")
frame_count = 0
frames_rgbd = []
frames = []
depth_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    depth_frame = getDepth(frame) # depth os each frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # parsing into RGB for further process
    frames.append(frame) # storing all the frames
    depth_frames.append(depth_frame)
    # parsing the frame and depth into a geometry representation of open3d
    color_o3d = o3d.geometry.Image(frame)
    depth_o3d = o3d.geometry.Image(depth_frame.astype(np.float32))

    # Creates rgbd structure based on the frame and its depth representation
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    frames_rgbd.append(rgbd)

cap.release()


#img = cv2.imread("link.png")

#output = getDepth(img)

plt.imshow(depth_frames[10])
plt.show()

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#color_o3d = o3d.geometry.Image(img)
#depth_o3d = o3d.geometry.Image(output.astype(np.float32))

# parameters of the camera... maybe use the metadata of the img,video...
fx_value = 525.0
fy_value = 525.0
cx_value = frames[0].shape[0] / 2
cy_value = frames[0].shape[1] / 2

# setting the parameters of the 3dd visualization
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=frames[0].shape[0], height=frames[0].shape[1], fx=fx_value, fy=fy_value, cx=cx_value, cy=cy_value)



vis = o3d.visualization.Visualizer()
vis.create_window()
extrinsic = np.identity(4) # test with np zeros or np ones to see what happens

# adds a new geometry based on the rgbd representation
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(frames_rgbd[0], intrinsic, extrinsic)
vis.add_geometry(pcd)


# function to uodate the visualization with a new rgbd
def update_point_cloud(vis, pcd, rgbd_image, intrinsic, extrinsic):
    new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)
    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

# shows the visualization and updates the rgbd to represent the cloud points
for rgbd_image in frames_rgbd:
    update_point_cloud(vis, pcd, rgbd_image, intrinsic, extrinsic)
    cv2.waitKey(5)

vis.destroy_window()
