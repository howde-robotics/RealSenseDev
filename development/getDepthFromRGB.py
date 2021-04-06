import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == "__main__":

    # Get the context
    ctx = rs.context()

    # Create the pipeline
    pipeline = rs.pipeline()

    # Configure the device
    cfg = rs.config()
    cfg.enable_all_streams()
    
    profile = pipeline.start(cfg)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 1000.000)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # There values are needed to calculate the mapping
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_min = 0.11 #meter
    depth_max = 10.0 #meter

    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
    color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))

    color_points = [
        [600.0, 200.0],
        [1200.0, 200.0],
        [600.0, 700.0],
        [1200.0, 700.0]
    ]

    depth_points = []
    for color_point in color_points:
        depth_point_ = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_frame.get_data(), depth_scale,
                    depth_min, depth_max,
                    depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_point)
        depth_points.append(depth_point_)

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    depth_points = np.array(depth_points)
    plt.imshow(depth)
    plt.scatter(depth_points[:,0],depth_points[:,1])
    plt.show()

    color_points = np.array(color_points)
    plt.imshow(color)
    plt.scatter(color_points[:,0],color_points[:,1])
    plt.show()

    center_idx = [0,0]
    center_idx[0] = int(((min(depth_points[:,0])+max(depth_points[:,0]))/2.0))
    center_idx[1] = int(((min(depth_points[:,1])+max(depth_points[:,1]))/2.0))
    # print(center_idx)
    # exit()
    center_dist = depth_frame.get_distance(center_idx[0], center_idx[1])
    print(center_dist)

    pipeline.stop()