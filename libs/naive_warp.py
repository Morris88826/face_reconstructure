import cv2
import argparse
import numpy as np

def naive_warp(source_img, landmark1, landmark2):

    # corresponding points (in 2D)


    # calculate transformation matrix
    M, mask = cv2.findHomography(landmark1[:, :2].astype(np.float32), landmark2[:, :2].astype(np.float32), cv2.RANSAC, 5.0)

    # warp image
    warped_img = cv2.warpPerspective(source_img, M, (source_img.shape[1], source_img.shape[0])) # adjust width and height of output
    
    return warped_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="dataset/val/n000029/0080_01.jpg", help='Path to the source directory')
    parser.add_argument('--target', type=str, default="image_bank/n000004.npy", help='Path to the image bank mask')
    args = parser.parse_args()

    source = args.source
    target = args.target

    out_path = "naive_warp.jpg"
    cv2.imwrite(out_path,naive_warp(source,target))