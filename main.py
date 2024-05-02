import os
import cv2
import tqdm
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from libs.plot import plot_landmarks
from libs.interpolation import interpolate_triangle_density
from libs.landmark import get_face_landmark, forward_warp, find_triangles, color_query
from libs.transformation.similarity import find_SimilarityMatrix
from libs.transformation.affine import find_AffineMatrix
from libs.naive_warp import naive_warp

def find_best_H(query_image, image_bank_dir):
    image_bank_paths = glob.glob(os.path.join(image_bank_dir, "*.npy"))
    query_image_lmk = get_face_landmark(query_image)

    min_error = 1e9
    best_H = None
    best_front_face_lmk = None
    for front_face_lmk in tqdm.tqdm(image_bank_paths):
        front_face_lmk = np.load(front_face_lmk)
        
        H = find_SimilarityMatrix(query_image_lmk, front_face_lmk)

        query_image_warped = forward_warp(query_image_lmk, H)
        error = np.mean(np.square(query_image_warped - front_face_lmk))

        if error < min_error:
            min_error = error
            best_H = H
            best_front_face_lmk = front_face_lmk

    return best_H, query_image_lmk, best_front_face_lmk

def get_dense_landmarks(query_image_lmk, density):
    triangles = find_triangles()
    augmented_lmks = query_image_lmk.copy()
    for triangle in triangles:
        A, B, C = augmented_lmks[triangle]
        points = interpolate_triangle_density(A, B, C, density)
        augmented_lmks = np.vstack((augmented_lmks, points))
    return augmented_lmks

def front_face_recovery(query_image, points, H):
    colors, valid_lmks = color_query(query_image, points) 

    homography_transformed_lmks = forward_warp(valid_lmks, H)
    max_x, max_y, _ = np.max(homography_transformed_lmks, axis=0).astype(int)
    min_x, min_y, _ = np.min(homography_transformed_lmks, axis=0).astype(int)
    out = np.zeros((max_y - min_y + 1, max_x - min_x + 1, 3), dtype=np.uint8)
    offset = np.array([min_x, min_y])
    out[homography_transformed_lmks[:, 1].astype(int) - offset[1], homography_transformed_lmks[:, 0].astype(int) - offset[0]] = colors

    return out
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Front Face reconstruction')
    parser.add_argument('--query_image', type=str, default='dataset/val/n000001/0007_01.jpg', help='Path to image')
    parser.add_argument('--image_bank', type=str, default='dataset/image_bank/samples', help='Path to image bank')
    parser.add_argument('--density', type=float, default=2, help='Density of points in triangle')
    args = parser.parse_args()

    query_image_path = args.query_image
    image_bank_dir = args.image_bank
    density = args.density

    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    best_H, query_image_lmk, best_front_face_lmk = find_best_H(query_image, image_bank_dir)

    ##### from sparse landmarks to dense landmarks
    dense_query_image_lmk = get_dense_landmarks(query_image_lmk, density)

    front_face = front_face_recovery(query_image, dense_query_image_lmk, best_H)

    ##### Plot the results 
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    warpped_landmarks = forward_warp(dense_query_image_lmk, best_H)

    axes[0,0].imshow(query_image)
    axes[0,0].set_title('Query Image')
    axes[0,0].axis('off')

    plot_landmarks(axes[0,1], query_image_lmk)
    axes[0,1].set_title('Query Image Landmarks')

    plot_landmarks(axes[0,2], best_front_face_lmk)
    axes[0,2].set_title('Best from the Image Bank')

    plot_landmarks(axes[1,0], warpped_landmarks)
    axes[1,0].set_title('Warpped Landmarks - Density: {}'.format(density))

    naive_result = naive_warp(query_image, query_image_lmk, best_front_face_lmk)
    axes[1,1].imshow(naive_result)
    axes[1,1].set_title('Recovered Front Face (Naive)')
    axes[1,1].axis('off')

    axes[1,2].imshow(front_face)
    axes[1,2].set_title('Recovered Front Face (Similairty Transformation)')
    axes[1,2].axis('off')

    verbose_dir  = "./verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)
    out_path = os.path.join(verbose_dir, 'front_face_recovery.png')
    fig.savefig(out_path)


