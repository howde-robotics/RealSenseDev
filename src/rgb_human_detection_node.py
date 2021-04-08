#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import pyrealsense2 as rs2
from darknet_ros_msgs.msg import BoundingBoxes
from dragoon_messages.msg import ObjectInfo
from dragoon_messages.msg import Objects
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Class for detected YOLO objects
class DetectedObject():
	def __init__(self):
		self.probability = 0.0
		self.id = 0
		self.Class = "None"
		self.xmin = 0
		self.xmax = 0
		self.ymin = 0
		self.ymax = 0

class rgb_human_detection_node():
	def __init__(self):
		# Needed for depth image visualization
		self.counter_depth = 0
		self.scat_depth = None

		# Needed for rgb image visualization
		self.counter_rgb = 0
		self.scat_rgb = None

		# Subscriber for YOLO topic
		self.bboxSub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.get_bbox)
		# Subscriber for depth image
		self.depthSub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.convert_depth_image)
		# Subscriber for RGB image - do NOT visualize depth and RGB simultaneously for now
		# self.rgbSub = rospy.Subscriber("/camera/color/image_raw", Image, self.show_color_image)

		# Publisher that will publish list of poses/object infos as an Objects message
		self.pub = rospy.Publisher('ObjectPoses', Objects, queue_size = 50)
		# Objects message to be published, consisting of poses/infos stored as ObjectInfo messages
		self.poseMsgs = Objects()
		# List of objects detected by YOLO in current frame
		self.obs = []

		# Hard-coded depth camera intrinsics
		self.depth_intrinsics = rs2.intrinsics()
		self.depth_intrinsics.width = 848
		self.depth_intrinsics.height = 480
		self.depth_intrinsics.ppx = 416.735
		self.depth_intrinsics.ppy = 250.361
		self.depth_intrinsics.fx = 422.726
		self.depth_intrinsics.fy = 422.726
		self.depth_intrinsics.model = rs2.distortion.brown_conrady
		self.depth_intrinsics.coeffs = [0, 0, 0, 0, 0]

		# Hard-coded RGB camera intrinsics
		self.color_intrinsics = rs2.intrinsics()
		self.color_intrinsics.width = 1280
		self.color_intrinsics.height = 720
		self.color_intrinsics.ppx = 629.794
		self.color_intrinsics.ppy = 373.026
		self.color_intrinsics.fx = 919.335
		self.color_intrinsics.fy = 919.664
		self.color_intrinsics.model = rs2.distortion.inverse_brown_conrady
		self.color_intrinsics.coeffs = [0, 0, 0, 0, 0]

		# Hard-coded extrinsics
		self.depth_to_color_extrin = rs2.extrinsics()
		self.depth_to_color_extrin.translation = [0.0147247, 0.000133323, 0.000350991]
		self.depth_to_color_extrin.rotation = [0.999929, 0.01182, -0.00152212, -0.0118143, 0.999923, 0.00370104, 0.00156575, -0.0036828, 0.999992]

		self.color_to_depth_extrin = rs2.extrinsics()
		self.color_to_depth_extrin.translation = [-0.0147247, 3.93503e-05, -0.000373552]
		self.color_to_depth_extrin.rotation = [0.999929, -0.0118143, 0.00156575, 0.01182, 0.999923, -0.0036828, -0.00152212, 0.00370104, 0.999992]

		# Depth camera scale (not used in current implementation)
		self.depth_scale = 0.0010000000475
		# Minimum detectable depth 
		self.depth_min = 0.11
		# Maximum detectable depth
		self.depth_max = 10.0

	# Projects a pixel from RGB frame to depth frame. Reference C++ code in link below:
	# http://docs.ros.org/en/kinetic/api/librealsense2/html/rsutil_8h_source.html#l00186
	# Difference is that I pass in converted depth from CvBridge instead of calculating depth using depth scale and raw depth data in line 213
	def project_color_to_depth(self, depth_image, from_pixel):
		to_pixel = [0.0,0.0]
		min_point = rs2.rs2_deproject_pixel_to_point(self.color_intrinsics, from_pixel, self.depth_min)
		min_transformed_point = rs2.rs2_transform_point_to_point(self.depth_to_color_extrin, min_point)
		start_pixel = rs2.rs2_project_point_to_pixel(self.depth_intrinsics, min_transformed_point)
		start_pixel = rs2.adjust_2D_point_to_boundary(start_pixel,self.depth_intrinsics.width, self.depth_intrinsics.height)

		max_point = rs2.rs2_deproject_pixel_to_point(self.color_intrinsics, from_pixel, self.depth_max)
		max_transformed_point = rs2.rs2_transform_point_to_point(self.depth_to_color_extrin, max_point)
		end_pixel = rs2.rs2_project_point_to_pixel(self.depth_intrinsics, max_transformed_point)
		end_pixel = rs2.adjust_2D_point_to_boundary(end_pixel,self.depth_intrinsics.width, self.depth_intrinsics.height)

		min_dist = -1.0
		p = [start_pixel[0], start_pixel[1]]
		next_pixel_in_line = rs2.next_pixel_in_line([start_pixel[0],start_pixel[1]], start_pixel, end_pixel)
		while rs2.is_pixel_in_line(p, start_pixel, end_pixel):
			depth = depth_image[int(p[1])*self.depth_intrinsics.width+int(p[0])]
			if depth == 0:
				p = rs2.next_pixel_in_line(p, start_pixel, end_pixel)
				continue   

			point = rs2.rs2_deproject_pixel_to_point(self.depth_intrinsics, p, depth)
			transformed_point = rs2.rs2_transform_point_to_point(self.color_to_depth_extrin, point)
			projected_pixel = rs2.rs2_project_point_to_pixel(self.color_intrinsics, transformed_point)

			new_dist = (projected_pixel[1]-from_pixel[1])**2+(projected_pixel[0]-from_pixel[0])**2
			if new_dist < min_dist or min_dist < 0:
				min_dist = new_dist
				to_pixel[0] = p[0]
				to_pixel[1] = p[1]

			p = rs2.next_pixel_in_line(p, start_pixel, end_pixel)
		return to_pixel

	# # Call this function to get depth intrinsics if camera is recalibrated
	# def imageDepthInfoCallback(cameraInfo):
	#     global intrinsics
	#     if intrinsics:
	#         return
	#     intrinsics = rs2.intrinsics()
	#     intrinsics.width = cameraInfo.width
	#     intrinsics.height = cameraInfo.height
	#     intrinsics.ppx = cameraInfo.K[2]
	#     intrinsics.ppy = cameraInfo.K[5]
	#     intrinsics.fx = cameraInfo.K[0]
	#     intrinsics.fy = cameraInfo.K[4]
	#     if cameraInfo.distortion_model == "plumb_bob":
	#         intrinsics.model = rs2.distortion.brown_conrady
	#     elif cameraInfo.distortion_model == "equidistant":
	#         intrinsics.model = rs2.distortion.kannala_brandt4
	#     intrinsics.coeffs = [i for i in cameraInfo.D]

	#     print(intrinsics)

	# Convert info from YOLO topic to a DetectedObject, add to self.obs
	def get_bbox(self, bounding_box_msg):
		self.obs = []
		for bbox in bounding_box_msg.bounding_boxes:
			YoloMsg = DetectedObject()
			YoloMsg.xmin = bbox.xmin
			YoloMsg.xmax = bbox.xmax
			YoloMsg.ymin = bbox.ymin
			YoloMsg.ymax = bbox.ymax
			YoloMsg.probability = bbox.probability
			YoloMsg.id = bbox.id
			YoloMsg.Class = bbox.Class
			self.obs.append(YoloMsg)

	# Convert raw depth data to actual depth, get actual depth info of YOLO bboxes and publish to 'ObjectPoses'
	def convert_depth_image(self, ros_image):
	    # try:
	    # if counter_depth % 5 == 0:
	    # Use cv_bridge() to convert the ROS image to OpenCV format
	    bridge = CvBridge()
	    # Convert the depth image using the default passthrough encoding
	    depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
	    # depth_image = bridge.imgmsg_to_cv2(ros_image, ros_image.encoding)

	    for obj in self.obs:
	    	# Get YOLO bbox corners
		    color_points = [
		        [float(obj.xmin), float(obj.ymin)],
		        [float(obj.xmax), float(obj.ymin)],
		        [float(obj.xmin), float(obj.ymax)],
		        [float(obj.xmax), float(obj.ymax)]
		    ]

		    # Convert each corner from color image to depth image
		    depth_points = []
		    for color_point in color_points:
		        depth_flattened = depth_image.flatten()
		        depth_point = self.project_color_to_depth(depth_flattened, color_point)
		        depth_points.append(depth_point)

		    depth_points = np.array(depth_points)

		    # Visualize bboxes in depth image
		    if self.scat_depth != None:
		        self.scat_depth.remove()
		    plt.imshow(depth_image)
		    self.scat_depth = plt.scatter(depth_points[:,0],depth_points[:,1])
		    plt.show()
		    plt.pause(0.00000000001)

		    # Get mins and maxes of depth image bbox
		    depth_xmin = min(depth_points[:,0])
		    depth_xmax = max(depth_points[:,0])
		    depth_ymin = min(depth_points[:,1])
		    depth_ymax = max(depth_points[:,1])

		    # Convert the depth image to a Numpy array
		    depth_array = np.array(depth_image, dtype=np.float32)

		    # Current implementation: get depth of center of bbox - need to fix
		    # center_idx = np.array(depth_array.shape) / 2
		    bbox_center_x = int((depth_xmax+depth_xmin)/2)
		    bbox_center_y = int((depth_ymax+depth_ymin)/2)
		    # print ('center depth:', depth_array[center_idx[0], center_idx[1]]/1000.00)
		    # print ('center depth:', depth_array[bbox_center_x, bbox_center_y]/1000.00)

		    if self.depth_intrinsics:
		        # center_depth = depth_array[center_idx[0], center_idx[1]]
		        center_depth = depth_array[bbox_center_y, bbox_center_x]
		        # result = rs2.rs2_deproject_pixel_to_point(depth_intrinsics, [center_idx[0], center_idx[1]], center_depth)

		        # Get 3D pose of point in depth frame
		        result = rs2.rs2_deproject_pixel_to_point(self.depth_intrinsics, [bbox_center_x, bbox_center_y], center_depth)

		        # New poseMsg with 3D pose info, probability, id, and class from YOLO
		        poseMsg = ObjectInfo()
		        poseMsg.pose.x = result[0]/1000.0
		        poseMsg.pose.y = result[1]/1000.0
		        poseMsg.pose.z = result[2]/1000.0
		        poseMsg.probability = obj.probability
		        poseMsg.id = obj.id
		        poseMsg.Class = obj.Class  

		        # Add new poseMsg to poseMsgs, Objects msg that will be published
		        self.poseMsgs.objects_info.append(poseMsg)

		# Publish Objects msg for frame once all YOLO objects are processed
	    self.pub.publish(self.poseMsgs)

	    # counter_depth += 1
	    # except CvBridgeError, e:
	        # print e

	# Visulize RGB stream with all bboxes
	def show_color_image(self, ros_image):
	    if self.counter_rgb % 10 == 0:
	        # stamp = ros_image.header.stamp
	        # time = stamp.secs + stamp.nsecs * 1e-9
	        bridge_color = CvBridge()
	        color_image = bridge_color.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")

	        for obj in self.obs:
		        if self.scat_rgb != None:
		            self.scat_rgb.remove()
		        plt.imshow(color_image)
		        self.scat_rgb = plt.scatter(np.array([obj.xmin, obj.xmin, obj.xmax, obj.xmax]),np.array([obj.ymin, obj.ymax, obj.ymin, obj.ymax]))
		        plt.show()
		        plt.pause(0.00000000001)

	    self.counter_rgb += 1

def main():
	rgbDetectionNode = rgb_human_detection_node()
	rospy.init_node('rgbHD',anonymous=True)

	# These two lines needed for continuous visualization
	plt.ion()
	plt.show()

	rospy.spin()

if __name__ == '__main__':
    main()
    
