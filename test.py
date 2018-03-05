import numpy as np
import os
import argparse
import caffe
from video import Video
from utils import center_crop_images
import skimage


class FeatureExtraction(object):
    """
    extract features for video frames.
    
    Parameters
    -----------------
    video: Video
    modelPrototxt: models architecture file
    modelFile: models snapshot
    featureLayer: which layer to be extracted as feature
    gpu_id: which gpu to use
    """

    def __init__(self, modelPrototxt, modelFile, featureLayer, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.net = caffe.Net(modelPrototxt, modelFile, caffe.TEST)
        data_shape = self.net.blobs['data'].data.shape
        self.batchsize = data_shape[0]
        self.height = data_shape[2]
        self.width = data_shape[3]

        self.featureLayer = featureLayer
        featureDim = self.net.blobs[featureLayer].data.shape
        print "featureDim:", featureDim

        transformer = caffe.io.Transformer({'data': data_shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))  # mean pixel
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer = transformer

    def __call__(self, video):
        if video.frame_group_len != self.batchsize:
            raise IOError(
                    ("FeatureExtraction error: video frame group len (%d) is not equal to prototxt batchsize (%d)"
                     % (self.video.frame_group_len, self.batchsize)))
            """
            warnings.warn(
                "FeatureExtractionWarning: "
                "video frame group len (%d) " % (self.video.frame_group_len) +
                "is not equal to prototxt batchsize (%d)" % (data_shape[0]) +
                "Change prototxt batchsize to video frame group len.",
                UserWarning)
            data_shape[0] = self.video.frame_group_len
            """
        for timestamps, frames in video: # frames are rgb channel-ordered
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

            yield timestamps, frames, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to videos")
    parser.add_argument("-f", "--filename", help="filename")
    parser.add_argument("-d", "--device", help="device id")
    parser.add_argument("-s", "--save", help="save")
    args = parser.parse_args()
    length = 3
    filename = args.filename
    print(filename)
    if not (os.path.isfile(args.save+filename[:-(length+1)]+'.binary')):
        video = Video(args.path+filename, frame_group_len=1, step=1)
        features = FeatureExtraction(modelPrototxt='./models/SENet.prototxt', modelFile='./models/SENet.caffemodel',
                      featureLayer='pool5/7x7_s1', gpu_id=int(args.device))
        feature_list = []
        for _, _, fea in features(video):
	    import pdb;pdb.set_trace()
            feature_list.append(np.squeeze(np.copy(fea)))
        feature = np.asarray(feature_list)
        feature.tofile(args.save+filename[:-(length+1)]+'.binary')
