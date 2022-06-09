'''
Purpose: visualize data from the dataframe.
- Write predictions on frames.
- Generate video from annotated frames.
'''
import glob
import os
import argparse
import cv2
import numpy as np
import pandas as pd


def write_predictions_on_frames(df):
    for idx, row in df.iterrows():
        fn = "{}.png".format(int(row['frame']))
        fp = os.path.join(frames_dir, fn)
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        if os.path.exists(fp):
            im = cv2.imread(fp)
            string = "({})".format(row['distance'])
            cv2.putText(im, string, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(fn, im)
            cv2.waitKey(0)
        else:
          print(fp)


def generate_video_from_frames():
    img_array = []
    imgs = glob.glob(os.path.join(frames_dir, '*.png'))
    for filename in imgs:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append([int(filename.split('/')[-1].split('.')[0]), img])
    img_array.sort()
    out = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), int(fps), size)
    for i in range(len(img_array)):
        out.write(img_array[i][1])
    out.release()


argparser = argparse.ArgumentParser(
    description='Generate annotations csv file from .txts')
argparser.add_argument('-d', '--data', help='input data csv file path')
argparser.add_argument(
    '-f', '--frames', help='input annotated video frames path')
argparser.add_argument('-fps', help="video frames per second")
argparser.add_argument('-r', '--results', help="output directory path")
args = argparser.parse_args()

# parse arguments
csvfile_path = args.data
frames_dir = args.frames
fps = args.fps
results_dir = args.results

# write predictions on frames
df = pd.read_csv(csvfile_path)
os.chdir(frames_dir)
write_predictions_on_frames(df)

# generate video from annotated frames
os.chdir(results_dir)
generate_video_from_frames()
