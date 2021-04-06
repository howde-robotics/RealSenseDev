import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Get the context
    ctx = rs.context()

    # Create the pipeline
    pipeline = rs.pipeline()

    # Configure the device
    cfg = rs.config()
    cfg.enable_all_streams()
    
    profile = pipeline.start(cfg)

    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

    # Set the exposure anytime during the operation
    # sensor.set_option(rs.option.exposure, 10.000)

    # Only the first frame is captured in manual exposure mode with an unknown exposure value!
    for _ in range(10):
        frames = pipeline.wait_for_frames()
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        print(depth)
        plt.imshow(depth)
        plt.show()

        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        center_idx = np.array(depth_array.shape) / 2
        print ('center depth:', depth_array[center_idx[0], center_idx[1]])

        pipeline.stop()