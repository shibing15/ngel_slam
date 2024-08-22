
#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray 
from orb_slam3_ros.msg import BA_info
import socket   
import sys
import json
import argparse
import numpy as np
import os
import yaml

def save_dict_to_yaml(dict_value, save_path):
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

def save_dict_to_json(dict_value, save_path):
    with open(save_path, 'w') as file:
        json.dump(dict_value, file)

class kf_subscriber:

    def __init__(self, save_path):
        self.count = 0
        self.kf_sub = rospy.Subscriber("/orb_slam3/BA_info", BA_info, self.callback)
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def callback(self, data):
        info = {}
        kf_list = []
        
        time_stamp = data.stamp.secs
        info['time'] = time_stamp
        info['countLBA'] = data.countLBA
        info['countGBA'] = data.countGBA
        info['countLoop'] = data.countLoop
        info['KFposes'] = {}

        kfs = data.KFposes
        for kf in kfs:
            frame_id = int(float(kf.header.frame_id))
            # print(time_stamp)

            px = kf.pose.position.x
            py = kf.pose.position.y
            pz = kf.pose.position.z

            ox = kf.pose.orientation.x
            oy = kf.pose.orientation.y
            oz = kf.pose.orientation.z
            ow = kf.pose.orientation.w

            kf_info = [px, py, pz, ox, oy, oz, ow]
            info['KFposes'][frame_id] = kf_info
            kf_list.append(frame_id)
            
        info['KFlist'] = kf_list
        file_path = os.path.join(self.save_path, "{}.json".format(self.count))
        save_dict_to_json(info, file_path)

        print(self.count, len(kf_list))
        self.count += 1


    
def listener(save_path):

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    

    rospy.init_node('kf_listener', anonymous=True)

    print('listner init!')

    kf_subscriber(save_path)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    save_path = 'keyframes'
    listener(save_path)