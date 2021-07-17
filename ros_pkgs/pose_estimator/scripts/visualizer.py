#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
import numpy as np

from sensor_msgs.msg import Image

import message_filters


class VisualizerNode:

    def __init__(self):
        rospy.init_node('visualizer')

        image_sub = message_filters.Subscriber('image', Image)
        pose_sub = message_filters.Subscriber('pose_estimation', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, pose_sub], 10, 5)
        self.ts.registerCallback(self.on_image_pose)

        # TODO: write pose estimation
        # self.pub = rospy.Publisher('pose_estimation_color', Image, queue_size=10)

        # self.br = cv_bridge.CvBridge()

        # self.pallete = np.array([
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 255],  # bicycle
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [255, 0, 0],  # car
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 255, 0],  # person
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0],
        #     [0, 0, 0]
        # ], dtype=np.uint8)


    def on_image_pose(self, image_msg : Image, segm_msg : Image): # TODO
        image = self.br.imgmsg_to_cv2(image_msg)
        # segm = self.br.imgmsg_to_cv2(segm_msg)

        # segm_color = self.pallete[segm]

        # segm_color_msg = self.br.cv2_to_imgmsg(segm_color, 'rgb8')
        # segm_color_msg.header = segm_msg.header

        # self.pub.publish(segm_color_msg)


    def spin(self):
        rospy.spin()


def main():
    node = VisualizerNode()
    node.spin()


if __name__ == '__main__':
    main()
