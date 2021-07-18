#!/usr/bin/env python3

import rospy
import cv_bridge
import numpy as np
import cv2
import matplotlib

import time

import torch
import torchvision
from torchvision.transforms import transforms as transforms

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
        # rospy.loginfo(f'Using device: {self.device}')

        # person detector module
        # TODO: try yolov3 or other models
        # rospy.loginfo('Loading detector model...')
        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.detector.classes = [0]  # only 'person' class
        self.detector.conf = 0.25  # confidence threshold

        # load pose model
        # TODO: workaround cfg
        # rospy.loginfo('Loading pose model...')
        cfg = update_config(cfg_path)  # TODO: add default value to cfg_path
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        # rospy.loginfo(f'Loading pose model weights from {weights_path}...')
        self.pose_model.load_state_dict(torch.load(weights_path))

        self.detector = self.detector.to(self.device)
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        # rospy.loginfo('Models loaded')

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
            width = bbox[2] - bbox[0]
            desirable_width = int((bbox[3] - bbox[1]) * (192 / 256))
            width_to_add = (desirable_width - width) // 2
            xmin = max(0, bbox[0]-width_to_add)
            xmax = xmin + desirable_width
            cropped_img = input_img[bbox[1]:bbox[3],xmin:xmax]
            # rospy.loginfo(f"crop image size: {cropped_img.shape}")
            cropped_img = cv2.resize(cropped_img, (192, 256))
            cropped_img = torch.Tensor(cropped_img)
            cropped_img = cropped_img.permute(2, 0, 1)
            cropped_batch[i] = cropped_img
            
        # rospy.loginfo("batched cropped images")
        # rospy.loginfo(cropped_batch.shape)    
        
        cropped_batch = cropped_batch.to(self.device)
        hm = self.pose_model(cropped_batch)

        # rospy.loginfo(hm.shape)

        hm = hm.cpu()
        bboxes = bboxes.cpu()
        pose_coords = []
        for i, bbox in enumerate(bboxes):
            pose_coords.append(heatmap_to_coord_simple(hm[i], bbox[:4])[0])
            # rospy.loginfo(heatmap_to_coord_simple(hm[i], bbox[:4])[0])

        return pose_coords


class KeypointRCNNModel:

    def __init__(self):
        self.transform =  transforms.Compose([
                transforms.ToTensor()
            ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
        self.model.to(self.device).eval()

    def _draw_keypoints(self, outputs, image):
        edges = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
            (12, 14), (14, 16), (5, 6)
        ]
        # the `outputs` is list which in-turn contains the dictionaries 
        for i in range(len(outputs[0]['keypoints'])):
            keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
            # proceed to draw the lines if the confidence score is above 0.9
            if outputs[0]['scores'][i] > 0.8:
                keypoints = keypoints[:, :].reshape(-1, 3)
                for p in range(keypoints.shape[0]):
                    # draw the keypoints
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                                3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    # uncomment the following lines if you want to put keypoint number
                    # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                for ie, e in enumerate(edges):
                    # get different colors for the edges
                    rgb = matplotlib.colors.hsv_to_rgb([
                        ie/float(len(edges)), 1.0, 1.0
                    ])
                    rgb = rgb*255
                    # join the keypoint pairs to draw the skeletal structure
                    # rospy.loginfo(f"\n\n!!!\ne = {e}\n{keypoints[e, 0][0]}, {keypoints[e, 1][0]}\n\n")
                    cv2.line(image, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])), (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])), tuple(rgb), 2)
            else:
                continue
        return image

    def process(self, image, orig_image):
        # rospy.loginfo(f"orig_numpy shape = {orig_image.shape}")

        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            output_image = self._draw_keypoints(outputs, orig_image)
        # rospy.loginfo(f"outputs: {outputs}")
        return output_image


class PoseEstimatorNode:

    def __init__(self):
        rospy.init_node('pose_estimator')

        # self.pose_estimator_model = PoseModel()
        self.rcnn_model = KeypointRCNNModel()

        self.sub = rospy.Subscriber('image_raw', Image, self.on_image, queue_size=10)
        self.pub = rospy.Publisher('pose_estimation', Image, queue_size=10)

        # rospy.loginfo("PoseEstimatorNode initiated")

        self.br = cv_bridge.CvBridge()

    def on_image(self, image_msg : Image):
        orig_image = self.br.imgmsg_to_cv2(image_msg)

        image = PoseEstimatorNode.preproc_cv2(orig_image)

        # orig_image = torch.Tensor(orig_image).permute(2, 0, 1).numpy()
        start_time = time.time()
        result = self.rcnn_model.process(image, orig_image)
        end_time = time.time() - start_time

        rospy.loginfo(f"time: {end_time}")

        # if self.pose_estimator_model.device == 'cuda':
        #     image = image.cuda()

        # pose_keypoints = self.pose_estimator_model.get_pose(image)

        # res_image = image.copy()

        # rospy.loginfo("got pose coordinates")
        # if len(pose_keypoints) > 0:
        #     for i in range(len(pose_keypoints)):
        #         rospy.loginfo(f"object: {i}\n")
        #         rospy.loginfo(pose_keypoints[i][0])
        #         rospy.loginfo(tuple(pose_keypoints[i][8].astype(int).tolist()))
        #         rospy.loginfo(tuple(pose_keypoints[i][9].astype(int).tolist()))
        #         rospy.loginfo("OK\n")
        #         color = (0, 255, 0)
        #         thickness = 3
        #         for j in range(16):
        #             res_image = cv2.circle(res_image, tuple(pose_keypoints[i][j].astype(int).tolist()), 5, color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][9].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][11].astype(int).tolist()), tuple(pose_keypoints[i][12].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][11].astype(int).tolist()), tuple(pose_keypoints[i][10].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][2].astype(int).tolist()), tuple(pose_keypoints[i][1].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][1].astype(int).tolist()), tuple(pose_keypoints[i][0].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][13].astype(int).tolist()), tuple(pose_keypoints[i][14].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][14].astype(int).tolist()), tuple(pose_keypoints[i][15].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][3].astype(int).tolist()), tuple(pose_keypoints[i][4].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][4].astype(int).tolist()), tuple(pose_keypoints[i][5].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][7].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][7].astype(int).tolist()), tuple(pose_keypoints[i][6].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][6].astype(int).tolist()), tuple(pose_keypoints[i][2].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][6].astype(int).tolist()), tuple(pose_keypoints[i][3].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][12].astype(int).tolist()), color, thickness)
                # res_image = cv2.line(res_image, tuple(pose_keypoints[i][8].astype(int).tolist()), tuple(pose_keypoints[i][13].astype(int).tolist()), color, thickness)

        segm_msg = self.br.cv2_to_imgmsg(result.astype(np.uint8))
        segm_msg.header = image_msg.header

        self.pub.publish(segm_msg)

    @staticmethod
    def preproc_cv2(image):
        image_tensor = torch.Tensor(image.copy()).float() / 255
        # mean = torch.Tensor([0.485, 0.456, 0.406])
        # std = torch.Tensor([0.229, 0.224, 0.225])

        # image_tensor = (image_tensor - mean) / std
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
