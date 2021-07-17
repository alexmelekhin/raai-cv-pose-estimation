#!/usr/bin/env python3

import rospy
import torch
import cv_bridge
import numpy as np
import cv2

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
        rospy.loginfo('Loading pose model...')
        cfg = update_config(cfg_path)  # TODO: add default value to cfg_path
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        rospy.loginfo(f'Loading pose model weights from {weights_path}...')
        self.pose_model.load_state_dict(torch.load(weights_path))

        self.detector = self.detector.to(self.device)
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        rospy.loginfo('Models loaded')

    # def get_pose(self, input_img):
    #     demo = SingleImageAlphaPose(args, cfg)
    #     # im_name = args.inputimg    # the path to the target image
    #     # image = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
    #     pose = demo.process("raw_frame", input_img)
    #     # img = demo.getImg()     # or you can just use: img = cv2.imread(image)
    #     img = demo.vis(input_img, pose)   # visulize the pose result

    #     return img

    def get_pose(self, input_img):

        bboxes = self.detector(input_img).xyxy[0].int()

        if bboxes.shape[0] == 0:
            return []
        
        cropped_batch = torch.zeros(bboxes.shape[0], 3, 256, 192)
        for i, bbox in enumerate(bboxes):
            # cropped_imgs.append(cv2.resize(input_img[bbox[1]:bbox[3],bbox[0]:bbox[2]], (256, 192)))
            cropped_img = (cv2.resize(input_img[bbox[1]:bbox[3],bbox[0]:bbox[2]], (192, 256)))
            cropped_img = torch.Tensor(cropped_img)
            cropped_img = cropped_img.permute(2, 0, 1)
            cropped_batch[i] = cropped_img
            
        rospy.loginfo("batched cropped images")
        rospy.loginfo(cropped_batch.shape)    
        
        cropped_batch = cropped_batch.to(self.device)
        hm = self.pose_model(cropped_batch)

        rospy.loginfo(hm.shape)

        hm = hm.cpu()
        bboxes = bboxes.cpu()
        pose_coords = []
        for i, bbox in enumerate(bboxes):
            pose_coords.append(heatmap_to_coord_simple(hm[i], bbox[:4])[0])
            rospy.loginfo(heatmap_to_coord_simple(hm[i], bbox[:4])[0])

        return pose_coords


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

        # image = PoseEstimatorNode.preproc_cv2(image)

        # if self.pose_estimator_model.device == 'cuda':
        #     image = image.cuda()

        pose_keypoints = self.pose_estimator_model.get_pose(image)

        res_image = image.copy()

        rospy.loginfo("got pose coordinates")
        if len(pose_keypoints) > 0:
            for i in range(len(pose_keypoints)):
                rospy.loginfo(f"object: {i}\n")
                rospy.loginfo(pose_keypoints[i][0])
                rospy.loginfo(tuple(pose_keypoints[i][8].astype(int).tolist()))
                rospy.loginfo(tuple(pose_keypoints[i][9].astype(int).tolist()))
                rospy.loginfo("OK\n")
                color = (0, 255, 0)
                thickness = 3
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][9].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][11].astype(int).tolist()), tuple(pose_keypoints[i][12].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][11].astype(int).tolist()), tuple(pose_keypoints[i][10].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][2].astype(int).tolist()), tuple(pose_keypoints[i][1].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][1].astype(int).tolist()), tuple(pose_keypoints[i][0].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][13].astype(int).tolist()), tuple(pose_keypoints[i][14].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][14].astype(int).tolist()), tuple(pose_keypoints[i][15].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][3].astype(int).tolist()), tuple(pose_keypoints[i][4].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][4].astype(int).tolist()), tuple(pose_keypoints[i][5].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][7].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][7].astype(int).tolist()), tuple(pose_keypoints[i][6].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][6].astype(int).tolist()), tuple(pose_keypoints[i][2].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][6].astype(int).tolist()), tuple(pose_keypoints[i][3].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][12].astype(int).tolist()), color, thickness)
                res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][13].astype(int).tolist()), color, thickness)


        # TODO: develop
        # line_color = (0, 255, 0)
        # line_thickness = 9
        # if len(pose_keypoints) > 0:
        #     for i in range(len(pose_keypoints)):
        #         res_image = cv2.line(image, pose_keypoints[8], pose_keypoints[9], line_color, line_thickness)

        segm_msg = self.br.cv2_to_imgmsg(res_image.astype(np.uint8))
        segm_msg.header = image_msg.header

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
    node = PoseEstimatorNode()
    node.spin()


if __name__ == '__main__':
    main()
