import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    from plot import plot_landmarks
    from transformation.affine import find_AffineMatrix
    from transformation.similarity import find_SimilarityMatrix
except ImportError:
    from libs.plot import plot_landmarks
    from libs.transformation.affine import find_AffineMatrix
    from libs.transformation.similarity import find_SimilarityMatrix

def get_face_landmark(image, model_asset_path='./checkpoints/models/face_landmarker_v2_with_blendshapes.task', detector=None):
    if detector is None:
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)

    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(mp_image)
        
    face_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in detection_result.face_landmarks[0]])
    face_landmarks[:, 0] *= image.shape[1]
    face_landmarks[:, 1] *= image.shape[0]

    return face_landmarks

def find_M(landmarks1, landmarks2, type='similarity'):
    assert landmarks1.shape == landmarks2.shape, 'landmarks1 and landmarks2 must have the same shape'
    assert type in ['similarity', 'affine'], 'type must be one of similarity or affine'
    
    if type == 'similarity':
        M = find_SimilarityMatrix(landmarks1, landmarks2)
    elif type == 'affine':
        M = find_AffineMatrix(landmarks1, landmarks2)
    else:
        raise ValueError('Invalid type')
    return M

def forward_warp(points, M):
    h_points = np.ones((points.shape[0], 4))
    h_points[:, :3] = points

    warped_points = np.dot(M, h_points.T).T
    warped_points = warped_points / warped_points[:, -1].reshape(-1, 1)
    return warped_points[:, :3]

def inverse_warp(points, M):
    M_inv = np.linalg.inv(M)
    return forward_warp(points, M_inv)

def find_triangles(edges=mp.solutions.face_mesh.FACEMESH_TESSELATION):
    graph = defaultdict(set) 
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    triangles = set()
    for u, v in edges:
        common_neighbors = graph[u].intersection(graph[v])
        for w in common_neighbors:
            triangle = tuple(sorted((u, v, w)))
            triangles.add(triangle)
    
    return np.array(list(triangles))

def color_query(image, points):
    colors = []
    valid_points = []
    for point in points:
        x, y = point[:2]
        x = int(x)
        y = int(y)
        
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue

        valid_points.append(point)
        colors.append(image[y, x])

    colors = np.array(colors)    
    valid_points = np.array(valid_points)   
    return colors, valid_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="dataset/val/n000001/0007_01.jpg", help='Path to the source image')
    parser.add_argument('--target', type=str, default="dataset/val/n000009/0014_01.jpg", help='Path to the target image (front face)')
    args = parser.parse_args()

    verbose_dir  = "./verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)

    source_path = args.source
    target_path = args.target

    source_image = cv2.imread(source_path)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    target_image = cv2.imread(target_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    source_landmarks = get_face_landmark(source_image)
    target_landmarks = get_face_landmark(target_image)

    M1 = find_M(source_landmarks, target_landmarks, type='similarity')
    M2 = find_M(source_landmarks, target_landmarks, type='affine')


    source_warped_similarity = forward_warp(source_landmarks, M1)
    source_warped_affine = forward_warp(source_landmarks, M2)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plot_landmarks(ax[0,0], source_landmarks, source_image)
    ax[0,0].set_title('Source')

    plot_landmarks(ax[0,1], source_warped_similarity)
    ax[0,1].set_title('Source Warped - Similarity')

    plot_landmarks(ax[1,0], target_landmarks, target_image)
    ax[1,0].set_title('Target')

    plot_landmarks(ax[1,1], source_warped_affine)
    ax[1,1].set_title('Source Warped - Affine')

    plt.savefig(os.path.join(verbose_dir, 'demo_landmarks.png'))

