# NGEL-SLAM

### [Paper](https://arxiv.org/pdf/2311.09525.pdf) | [Video](https://www.bilibili.com/video/BV1W94y1G7gb/?share_source=copy_web&vd_source=e960221a1c45f15df36f53b3dce89e1c) | [Project Page]()

> NGEL-SLAM: Neural Implicit Representation-based Global Consistent Low-Latency SLAM System <br />
> Yunxuan Mao, Xuan Yu, Kai Wang, Yue Wang, Rong Xiong, Yiyi Liao<br />
> **Winner of ICRA 2024 Best Paper Award in Robot Vision**

<p align="center">
  <a href="">
    <img src="./media/teaser.png" alt="Logo" width="60%">
  </a>
</p>


## Installation

Please follow the instructions below to install the repo and dependencies.

```bash
mkdir catkin_ws && cd catkin_ws
mkdir src && cd src
git clone https://github.com/YunxuanMao/ngel_slam.git
cd ..
catkin_make
```
<!-- See [orb-slam3-ros](https://github.com/thien94/orb_slam3_ros) for more detail. -->
ORB-SLAM-ROS3 modified by me can be downloaded here: [google dirve](https://drive.google.com/drive/folders/1RvMYtInNbKuP8XBV2_Z4iEpCbDixKxLE?usp=drive_link) or [baidu netdisk](https://pan.baidu.com/s/1SUxY7pP_9de5kmw5G_7qYA) (Passward: di2k)

### Install the environment

```bash
conda create -n ngel python=3.8
conda activate ngel

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt

cd third_party/kaolin
python setup.py develop

cd third_party/kaolin-wisp
pip install -r requirements.txt
pip install -r requirements_app.txt
python setup.py develop
```

<!-- What's more, [kaolin-wisp](https://kaolin-wisp.readthedocs.io/en/latest/pages/install.html) should be installed. -->
<!-- ORB-SLAM-ROS3 modified by me can be downloaded here: [google dirve](https://drive.google.com/drive/folders/1RvMYtInNbKuP8XBV2_Z4iEpCbDixKxLE?usp=drive_link) or [baidu netdisk](https://pan.baidu.com/s/1SUxY7pP_9de5kmw5G_7qYA) (Passward: di2k) -->


## Data Preparation

You should put your data in `data` folder follow [NICE-SLAM](https://github.com/cvg/nice-slam) and generate a rosbag for ORB-SLAM3

```
python write_bag.py --input_folder '{PATH_TO_INPUT_FOLDER}' --output '{PATH_TO_ROSBAG}' --frame_id 'FRAME_ID_TO_DATA'
```
You should change the intrinsics manually in `orb_utils/write_bag.py`.

## Run

### ROS Communication
This version is for those who can run ORB and NGEL on the same device and at the same time.

You should first start the ORB-SLAM3-ROS, and then using code below

```
python main.py --config '{PATH_TO_CONFIG}'  --input_folder '{PATH_TO_INPUT_FOLDER}' --output '{PATH_TO_OUTPUT}' 
```

### NGEL with JSON frame infos
This version is for those who cannot run ORB and NGEL at the same time. 

You should first start the ORB-SLAM3-ROS and the `orb_utils/BA_subscriber.py` to save the keyframe information for every timestamps (You should change the save path manually in the script to `{PATH_TO_INPUT_FOLDER}/keyframes`).
```
python main_json.py --config '{PATH_TO_CONFIG}'  --input_folder '{PATH_TO_INPUT_FOLDER}' --output '{PATH_TO_OUTPUT}' 
```


## Citation

If you find our code or paper useful for your research, please consider citing:

```
@article{mao2023ngel,
  title={Ngel-slam: Neural implicit representation-based global consistent low-latency slam system},
  author={Mao, Yunxuan and Yu, Xuan and Wang, Kai and Wang, Yue and Xiong, Rong and Liao, Yiyi},
  journal={arXiv preprint arXiv:2311.09525},
  year={2023}
}
```

```
@inproceedings{mao2024ngel,
  title={Ngel-slam: Neural implicit representation-based global consistent low-latency slam system},
  author={Mao, Yunxuan and Yu, Xuan and Zhang, Zhuqing and Wang, Kai and Wang, Yue and Xiong, Rong and Liao, Yiyi},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6952--6958},
  year={2024},
  organization={IEEE}
}
```

For large scale mapping work, you can refer to [NF-Atlas](https://github.com/yuxuan1206/NF-Atlas).

## Acknowledge
Thanks for the source code of [orb-slam3-ros](https://github.com/thien94/orb_slam3_ros), [kaolin](https://kaolin.readthedocs.io/en/latest/) and [kaolin-wisp](https://kaolin-wisp.readthedocs.io/en/latest/pages/install.html).