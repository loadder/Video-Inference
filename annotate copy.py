import json
import os
import argparse
import csv
from copy import deepcopy
import cv2
domain = "oxy45khzj.bkt.clouddn.com"
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset")
parser.add_argument("-c", "--classes")
parser.add_argument("-s", "--save")
default_dict = {
    "url": "",
    "type": "video",
    "source_url": "",
    "ops": "",
    "Video_info": {
        "duration": 0,
        "resolution": ""
     },
    "label":
    {
            "class": {},
            "detect": {},
            "facecluster": "",
            "action-detect": {
                 "actions": [
                  {
                    "label": "",
                    "segment": []
                  }
                  ]
             }
        }
}
args = parser.parse_args()
annotations_list = ["activity_net.v1-3.min.json"]
w = open(args.save, 'w')
for annotations in annotations_list:
    annotations = open(annotations, 'r')
    anno = json.load(annotations)['database']
    whole_dict = dict()
    for id, data in anno.items():
        video_id = "v_" + "-"*(13-len(id)-2) + id
        print(video_id)
        file_dict = deepcopy(default_dict)
        file_dict["url"] = domain + "/" + video_id + ".mp4"
        video_dir = "/workspace/untrimmed-data-xcm/ActivityNet/videos" + "/" + video_id + ".mp4"
        '''
        video = cv2.VideoCapture(video_dir)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = length * 1.0 / fps
        '''
        file_dict["video_info"] = {"duration": data['duration'],
                                   "resolution": data['resolution']}
        action_anno = data['annotations']
        action_des = list(map(lambda x: x['label'], action_anno))
        action_seg = list(map(lambda x: x['segment'], action_anno))
        action_seg = list(map(lambda x: list(map(float, x.split(" ", 1))), action_seg))
        file_dict["label"]["action-detect"]["actions"] = [{
                "label": action_des,
                "segment": action_seg
            }]
        file_json = json.dumps(file_dict, indent=4)
        w.writelines(file_json)
        w.writelines('\n')
w.close()
