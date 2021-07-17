#!/usr/bin/env python3

import rospy
import torch
import cv_bridge
import numpy as np

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import heatmap_to_coord_simple

from sensor_msgs.msg import Image


class PoseModel:

    def __init__(self, cfg_path='/catkin_ws/configs/fastpose_mpii.yaml', 
                       weights_path='/catkin_ws/weights/model_best.pth',
                       device=''):

        self.device = device
        if (self.device == '') or ((self.device != 'cpu') and (self.device != 'cuda')):
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                device = 'cpu'
        rospy.loginfo(f'Using device: {self.device}')

        # person detector module
        # TODO: try yolov3 or other models
        rospy.loginfo('Loading detector model...')
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.detector.classes = [0]  # only 'person' class
        self.detector.conf = 0.25  # confidence threshold

        # load pose model
        # TODO: workaround cfg
        # rospy.loginfo('Loading pose model...')
        # cfg = update_config(cfg_path)  # TODO: add default value to cfg_path
        # self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        # rospy.loginfo(f'Loading pose model weights from {weights_path}...')
        # self.pose_model.load_state_dict(torch.load(weights_path, map_location=device))

        self.detector = self.detector.to(self.device)
        # self.pose_model = self.pose_model.to(device)
        # self.pose_model.eval()
        rospy.loginfo('Models loaded')

    def get_pose(self, input_img):

        rospy.loginfo(type(input_img))
        rospy.loginfo(input_img.shape)

        bboxes = self.detector(input_img[0]).xyxy[0].int()

        # cropped_imgs = torch.zeros(bboxes.shape[0])
        # for i, bbox in enumerate(bboxes):
        #     cropped_imgs[i] = input_img[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        rospy.loginfo("meow")
        
        # hm = self.pose_model(cropped_images)

        # pose_coords = heatmap_to_coord_simple(hm, bboxes)

        # return pose_coords


class PoseEstimatorNode:

    def __init__(self):
        rospy.init_node('pose_estimator')

        self.pose_estimator_model = PoseModel()

        self.sub = rospy.Subscriber('image_raw', Image, self.on_image, queue_size=10)
        self.pub = rospy.Publisher('pose_estimation', Image, queue_size=10)

        rospy.loginfo("PoseEstimatorNode initiated")

        self.br = cv_bridge.CvBridge()

    def on_image(self, image_msg : Image):
        image = self.br.imgmsg_to_cv2(image_msg)

        image = PoseEstimatorNode.preproc_cv2(image)

        if self.pose_estimator_model.device == 'cuda':
            image = image.cuda()

        pose_keypoints = self.pose_estimator_model.get_pose(image)

        # TODO: develop

        # segm_msg = self.br.cv2_to_imgmsg(segm.cpu().numpy().astype(np.uint8), 'mono8')
        # segm_msg.header = image_msg.header

        self.pub.publish(image_msg)

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
    node = PoseEstimatorNode()
    node.spin()


if __name__ == '__main__':
    main()
