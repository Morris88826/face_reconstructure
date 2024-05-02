import numpy as np
from math import cos, sin
from scipy.optimize import least_squares

def transformation_matrix(params):
    s, x, y, z, a, b, c = params
    D = np.array([
        [s, 0, 0, x],
        [0, s, 0, y],
        [0, 0, s, z],
        [0, 0, 0, 1]
    ])
    
    R_x = np.array([
        [1, 0, 0, 0],
        [0, cos(a), -sin(a), 0],
        [0, sin(a), cos(a), 0],
        [0, 0, 0, 1]
    ])
    
    R_y = np.array([
        [cos(b), 0, sin(b), 0],
        [0, 1, 0, 0],
        [-sin(b), 0, cos(b), 0],
        [0, 0, 0, 1]
    ])
    
    R_z = np.array([
        [cos(c), -sin(c), 0, 0],
        [sin(c), cos(c), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    H = R_z @ R_y @ R_x @ D

    # Return combined transformation
    return H

def residual(params, landmarks1, landmarks2):
    M = transformation_matrix(params)
    errors = []

    # Compute the error between each transformed P and Q
    for P, Q in zip(landmarks1, landmarks2):
        transformed_P = M @ P
        errors.append(transformed_P - Q)

    # Flatten errors for the optimizer
    return np.concatenate(errors)

def normalize(landmarks):
    landmarks = np.array(landmarks)
    offset = np.mean(landmarks, axis=0)
    landmarks = landmarks - offset
    scaling = np.max(np.abs(landmarks), axis=0)
    landmarks = landmarks / scaling
    return landmarks, (offset, scaling)

def find_SimilarityMatrix(landmarks1, landmarks2, verbose=False):
    normalized_face_landmarks1, (offset1, scaling1) = normalize(landmarks1)
    normalized_face_landmarks2, _ = normalize(landmarks2)
    
    h_landmarks1 = np.hstack((normalized_face_landmarks1, np.ones((normalized_face_landmarks1.shape[0], 1))))
    h_landmarks2 = np.hstack((normalized_face_landmarks2, np.ones((normalized_face_landmarks2.shape[0], 1))))

    # Find the optimal transformation matrix
    initial_guess = [1, 0, 0, 0, 0, 0, 0] # s, x, y, z, a, b, c
    result = least_squares(lambda params: residual(params, h_landmarks1, h_landmarks2), initial_guess)
    H = transformation_matrix(result.x)

    tH = np.array([
        [1, 0, 0, -offset1[0]],
        [0, 1, 0, -offset1[1]],
        [0, 0, 1, -offset1[2]],
        [0, 0, 0, 1]
    ])

    sH = np.array([
        [1/scaling1[0], 0, 0, 0],
        [0, 1/scaling1[1], 0, 0],
        [0, 0, 1/scaling1[2], 0],
        [0, 0, 0, 1]
    ])

    H =  np.linalg.inv(tH) @ np.linalg.inv(sH) @ H @ sH @ tH

    if verbose:
        return H, result.x
    else:
        return H