import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import pyrealsense2 as rs
from imutils.video import FPS


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]


def load_image_frame(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.imshow('image', img_flo[:, :, [0,1,2]]/255.0)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    fps = None

    with torch.no_grad():
        fps = FPS().start()

        try:
            image1 = None
            image2 = None
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                im = np.asanyarray(color_frame.get_data())[:,:,::-1].astype(np.uint8)
                img = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)


                if image2 is None:
                    image1 = img
                else:
                    image1 = image2

                image2 = img

                flow_low, flow_up = model(image1, image2, iters=args.iters, test_mode=True)
                fps.update()
                fps.stop()
                print(f"FPS: {fps.fps():.2f}")
                viz(image1, flow_up)




                # check for escape key
                key = cv2.waitKey(1)
                if key == 27 or key == 1048603:
                    break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--iters', default=20, type=int, help='number of refinement iters (longer=better prediction, but more latency)')
    args = parser.parse_args()

    demo(args)
