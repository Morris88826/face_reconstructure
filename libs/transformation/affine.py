import numpy as np

def find_AffineMatrix(landmarks1, landmarks2):
    num_points = landmarks1.shape[0]
    # Compute homography
    A = []
    for i in range(num_points):
        x1, y1, z1 = landmarks1[i]
        x2, y2, z2 = landmarks2[i]
        # 3d
        A.append([x1, y1, z1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -x2])
        A.append([0, 0, 0, 0, x1, y1, z1, 1, 0, 0, 0, 0, -y2])
        A.append([0, 0, 0, 0, 0, 0, 0, 0, x1, y1, z1, 1, -z2])
    
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = np.eye(4)
    H[0] = V[-1][:4]
    H[1] = V[-1][4:8]
    H[2] = V[-1][8:12]
    H[3,-1] = V[-1][-1]
    H = H / H[-1, -1]
    return H