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


    # Get the sensor once at the beginning. (Sensor index: 1)
    # sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

    # # Set the exposure anytime during the operation
    # #sensor.set_option(rs.option.exposure, 1000.000)

    # # Only the first frame is captured in manual exposure mode with an unknown exposure value!
    # for _ in range(10):
    #     frames = pipeline.wait_for_frames()
    #     depth = np.asanyarray(frames.get_depth_frame().get_data())
    #     color = np.asanyarray(frames.get_color_frame().get_data())
    #     np.save("img15_depth", depth)
    #     np.save("img15_color", color)
    #     # exit()

    #     test = np.load("img15_depth.npy")
    #     test_color = np.load("img15_color.npy")
    #     plt.imshow(test)
    #     plt.show()

    #     plt.imshow(test_color)
    #     plt.show()
    #     # print(depth)
    #     # plt.imshow(depth)
    #     # plt.imsave("test.jpg", depth)
    #     # depth2 = depth*float(255)
    #     # depth2 = depth.astype('uint8')
    #     # cv2.imwrite("test.jpg", depth2)

    #     # cv2.imshow("test.jpg")
    #     # plt.show()
    pipeline.stop()