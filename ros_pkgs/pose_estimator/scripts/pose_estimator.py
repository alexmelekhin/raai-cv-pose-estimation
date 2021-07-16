#!/usr/bin/env python3

import rospy
import torch
import cv_bridge
import numpy as np

from alphapose.models import builder
from alphapose.utils.config import update_config

from sensor_msgs.msg import Image


class PoseEstimatorNode:

    def __init__(self, cfg_path):
        rospy.init_node('pose_estimator')

        # person detector module
        # TODO: try yolov3 or other models
        rospy.loginfo('Loading detector model...')
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.detector.classes = [0]  # only 'person' class
        self.detector.conf = 0.25  # confidence threshold

        # load pose model
        # TODO: workaround cfg
        rospy.loginfo('Loading pose model...')
        cfg = update_config(cfg_path)  # TODO: add default value to cfg_path
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        if torch.cuda.is_available():
            self.pose_model = self.model.cuda()
        rospy.loginfo('Models loaded')

        self.sub = rospy.Subscriber('image', Image, self.on_image, queue_size=10)
        self.pub = rospy.Publisher('pose_estimation', Image, queue_size=10)

        self.br = cv_bridge.CvBridge()

    def on_image(self, image_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg)

        # TODO: develop

        # batch = PoseEstimatorNode.preproc_cv2(image)
        # if torch.cuda.is_available():
        #     batch = batch.cuda()
        # logits = self.model(batch)

        # probs = torch.softmax(logits['aux'][0], 0)
        # segm = probs.argmax(dim=0) * (probs.max(dim=0).values > 0.5)

        # segm_msg = self.br.cv2_to_imgmsg(segm.cpu().numpy().astype(np.uint8), 'mono8')
        # segm_msg.header = image_msg.header

        self.pub.publish(segm_msg)

    @staticmethod
    def preproc_cv2(image):
        image_tensor = torch.Tensor(image.copy()).float() / 255
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.permute(2, 0, 1)

        batch = image_tensor.unsqueeze(0)

        return batch


    def spin(self):
        rospy.spin()


def main():
    pass