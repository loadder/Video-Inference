import json
import os
import argparse
import csv
from copy import deepcopy
import cv2
domain = "oy1v528iz.bkt.clouddn.com"
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
class_description = open(args.classes).readlines()
class_description = dict(zip(list(map(lambda x:x.split(' ', 1)[0], class_description)),
                                  list(map(lambda x: x.split(' ', 1)[1].rstrip(), class_description))))
annotations_list = ["./Charades/Charades_v1_test.csv", "./Charades/Charades_v1_train.csv"]
w = open(args.save, 'w')
for annotations in annotations_list:
    annotations = open(annotations)
    anno = csv.DictReader(annotations)
    whole_dict = dict()
    for row in anno:
        video_id = row['id']
	print(video_id)
        file_dict = deepcopy(default_dict)
        file_dict["url"] = domain + "/" + args.dataset + "/" + video_id + ".mp4"
        video_dir = "/workspace/untrimmed-data-xcm/videos/" + args.dataset + "/" + video_id + ".mp4"
        video = cv2.VideoCapture(video_dir)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = length * 1.0 / fps
        file_dict["video_info"] = {"duration": duration,
                                   "resolution": str(width) + "x" + str(height)}
	if row['actions'] != '':
	        action_des = list(map(lambda x: class_description[x.split(" ", 1)[0]], row['actions'].split(';')))
	        action_seg = list(map(lambda x: x.split(" ", 1)[1], row['actions'].split(';')))
		action_seg = list(map(lambda x: list(map(float, x.split(" ", 1))), action_seg))
	        file_dict["label"]["action-detect"]["actions"] = [{
	            "label": action_des,
	            "segment": action_seg
        	}]
	file_json = json.dumps(file_dict, indent=4)
    	w.writelines(file_json)
        w.writelines('\n')
w.close()
