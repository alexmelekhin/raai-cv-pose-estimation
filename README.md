# Pose estimation project for RAAI summer school's CV track

Baseline forr this project is entirely based on [AlphaPose multi-person pose estimator](https://github.com/MVIG-SJTU/AlphaPose).

## Instructions

- Clone AlphaPose into `baselines` directory: 
    ```shell
    cd baselines/
    git clone https://github.com/MVIG-SJTU/AlphaPose.git
    git pull origin pull/592/head
    ```

- **Important:** After building image and starting container first execute `baselines/AlphaPose/additional_setup.sh` script **inside the container**.

- Download the object detection model manually: **yolov3-spp.weights**([Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC) | [Baidu pan](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)). Place it into `baselines/AlphaPose/detector/yolo/data`.

- Configure paths to datasets in `start.sh` script.

- Other info in [AlphaPose documentation](https://github.com/MVIG-SJTU/AlphaPose/blob/master/README.md)
