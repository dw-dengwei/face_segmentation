import os
import sys
import argparse
import numpy as np
from glob import glob
from PIL import Image
import surgery
from tqdm import tqdm
from pathlib import Path
import caffe
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class Data(Dataset):
    def __init__(self, fps, input_root):
        super().__init__()
        self.fps = fps
        self.input_root = input_root

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        fp = os.path.join(self.input_root, self.fps[index])
        im = Image.open(fp)
        ori_size = im.size
        im = im.resize((500, 500))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        return im, in_, fp, ori_size


def segment(im, in_, image_path, ori_size, args):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    # im = Image.open(image_path)
    # ori_size = im.size
    # im = im.resize((500, 500))
    # in_ = np.array(im, dtype=np.float32)
    # in_ = in_[:,:,::-1]
    # in_ -= np.array((104.00698793,116.66876762,122.67891434))
    # in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(len(in_), *in_[0].shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data.argmax(axis=1)

    # save result to .npy file
    # file_name, file_ext = os.path.splitext(input.split("/"))
    for i in range(len(image_path)):
        file_name = Path(image_path[i]).stem
        output_file_path = os.path.join(args.output, file_name + '.png')

        # np.save(output_file_path, out)
        im_tmp = np.array(im[i].resize(ori_size[i]))
        out_tmp = np.array(Image.fromarray(out[i, ...].astype(np.uint8)).resize(ori_size[i]))
        out_tmp[out_tmp > 0] = 1
        out_tmp = np.repeat(np.expand_dims(out_tmp, 2), 3, axis=2)
        im_tmp *= out_tmp
        im_tmp = Image.fromarray(im_tmp)
        # Image.fromarray(out_tmp).save(output_file_path)
        im_tmp.save(output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Segmentation')
    parser.add_argument('--input', type=str, default='./input', help='input image or directory path')
    parser.add_argument('--output', type=str, default='./output', help="output directory path")
    parser.add_argument('--start', type=int, default=1, help='input image or directory path')
    parser.add_argument('--end', type=int, default=5000, help="output directory path")
    parser.add_argument('--gpu', type=int, default=0, help="output directory path")
    args = parser.parse_args()

    # if you want to run on gpu, uncomment these 2 lines:
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    # load net
    net = caffe.Net('data/face_seg_fcn8s_deploy.prototxt', 'data/face_seg_fcn8s.caffemodel', caffe.TEST)
    for idx in range(args.start, 1 + args.end):
        args.input = f'/home/dw/data/vgg/aligned/{idx}/'
        args.output = f'/home/dw/data/vgg/face/{idx}/'
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        dataset = Data(
            glob(args.input + '*.jpg'), args.input
        )
        dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=8, pin_memory=True, shuffle=False, collate_fn=lambda x:x)
        for batch in tqdm(dataloader, file=sys.stdout, desc=f'{idx}'):
            batch = list(zip(*batch))
            im, in_, fp, ori_size = batch[0], batch[1], batch[2], batch[3]
            segment(im, in_, fp, ori_size, args)

