#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

xmin = 600
xmax = 1200
ymin = 200
ymax = 700

def convert_depth_image(ros_image):
    bridge = CvBridge()
     # Use cv_bridge() to convert the ROS image to OpenCV format
    try:
     #Convert the depth image using the default passthrough encoding
        depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
        plt.imshow(depth_image)
        plt.scatter(np.array([xmin, xmin, xmax, xmax]),np.array([ymin, ymax, ymin, ymax]))
        plt.show()
        plt.close()
        
        depth_array = np.array(depth_image, dtype=np.float32)
        center_idx = np.array(depth_array.shape) / 2
        print ('center depth:', depth_array[center_idx[0], center_idx[1]]/1000.00)

    except CvBridgeError, e:
        print e
     #Convert the depth image to a Numpy array

def show_color_image(ros_image):
    bridge_color = CvBridge()
    color_image = bridge_color.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
    plt.imshow(color_image)
    plt.scatter(np.array([xmin, xmin, xmax, xmax]),np.array([ymin, ymax, ymin, ymax]))
    plt.show()
    plt.close()

def pixel2depth():
    rospy.init_node('pixel2depth',anonymous=True)
    # rospy.Subscriber("/camera/depth/image_rect_raw", Image,callback=convert_depth_image, queue_size=1)
    # rospy.Subscriber("/camera/color/image_raw", Image, show_color_image, (xmin, xmax, ymin, ymax))
    rospy.Subscriber("/camera/color/image_raw", Image, show_color_image)
    # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, convert_depth_image, (xmin, xmax, ymin, ymax))
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, convert_depth_image)
    rospy.spin()

if __name__ == '__main__':
    pixel2depth()
    