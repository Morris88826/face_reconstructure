import matplotlib.pyplot as plt
    
def plot_landmarks(ax, landmarks, image=None):
    if image is not None:
        ax.imshow(image)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', label='landmarks', s=3)

    if image is not None:
        ax.axis('off')
    else:
        ax.axis('equal')
        ax.invert_yaxis()
