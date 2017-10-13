import numpy as np
import caffe
from video import Video
from utils import center_crop_images
import skimage
import cv2
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--suffix", help="the suffix of video")
parser.add_argument("-p", "--path", help="path to videos")
args = parser.parse_args()

class FeatureExtraction(object):
    """
    extract features for video frames.
    
    Parameters
    -----------------
    video: Video
    modelPrototxt: model architecture file
    modelFile: model snapshot
    featureLayer: which layer to be extracted as feature
    gpu_id: which gpu to use
    """

    def __init__(self, video, modelPrototxt='./models/SENet.prototxt', modelFile='./models/SENet.caffemodel',
                 featureLayer='pool5/7x7_s1', gpu_id=0):
        self.video = video

        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.net = caffe.Net(modelPrototxt, modelFile, caffe.TEST)
        data_shape = self.net.blobs['data'].data.shape
        self.height = data_shape[2]
        self.width = data_shape[3]
        self.batchsize = data_shape[0]
        if self.video.frame_group_len != self.batchsize:
            raise IOError(
					("FeatureExtraction error: video frame group len (%d) is not equal to prototxt batchsize (%d)"
					 % (self.video.frame_group_len, self.batchsize)))
        self.featureLayer = featureLayer
        featureDim = self.net.blobs[featureLayer].data.shape
        print "featureDim:", featureDim

        transformer = caffe.io.Transformer({'data': data_shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))  # mean pixel
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer = transformer

    def __call__(self):
        for timestamps, frames in self.video: # frames are rgb channel-ordered
            center_frames = center_crop_images(frames,(self.height, self.width))
            #### TO DO: add other crop functions, corner crop, flip, etc.
            im_group = np.empty((self.batchsize,3,self.height,self.width), dtype=np.float32)

            for ix, img in enumerate(center_frames):
                #### TO DO: preprocess multi images at once.
                img = skimage.img_as_float(img).astype(np.float32)
                img_preprocess = self.transformer.preprocess('data', img)
                im_group[ix] = img_preprocess

            self.net.blobs['data'].data[...] = im_group
            self.net.forward()
            features = self.net.blobs[self.featureLayer].data[...].reshape(self.batchsize, -1)

            yield features


if __name__ == "__main__":
    suffix = args.suffix
    length = len(suffix)
    files = os.listdir(args.path)
    os.mkdir(args.path+'/features')
    for filename in files:
        print(filename)
        if filename[-length:] == suffix:
            video = Video(filename, frame_group_len=1)
            features = FeatureExtraction(video, modelPrototxt='./models/SENet.prototxt', modelFile='./models/SENet.caffemodel',
                         featureLayer='pool5/7x7_s1', gpu_id=0)
            feature_list = []
            for fea in features():
                feature_list.append(np.squeeze(fea))
            feature = np.asarray(feature_list)
            np.save(args.path+"/features/"+filename[:-(length+1)]+'.npy', feature)
