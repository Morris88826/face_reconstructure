import os
import cv2
import tqdm
import json
import glob
import argparse
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from libs.landmark import get_face_landmark
from libs.transformation.similarity import find_SimilarityMatrix


def find_front_face(image, samples_path="dataset/image_bank/samples", threshold=10, detector=None):
    landmark = get_face_landmark(image, detector=detector)
    rxs = []
    rys = []
    rzs = []
    for front_face in glob.glob(samples_path + "/*.npy"):
        front_face_lmks = np.load(front_face)
        
        _, params = find_SimilarityMatrix(landmark, front_face_lmks, verbose=True)
        # s, tx, ty, tz, rx, ry, rz
        _, _, _, _, rx, ry, rz = params
        rx = np.rad2deg(rx)
        ry = np.rad2deg(ry)
        rz = np.rad2deg(rz)

        rxs.append(rx)
        rys.append(ry)
        rzs.append(rz)

    mean_rx = np.mean(rxs)
    mean_ry = np.mean(rys)
    mean_rz = np.mean(rzs)

    if abs(mean_rx) > threshold or abs(mean_ry) > threshold or abs(mean_rz) > threshold:
        return False
    else:
        return True
    
def find_front_faces(subject_dir, num_samples=10, detector=None):
    images = glob.glob(os.path.join(subject_dir, "*.jpg"))
    print(f"Found {len(images)} images in {subject_dir}")

    front_faces_path = []
    for image_path in tqdm.tqdm(images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if find_front_face(image, detector=detector):
            front_faces_path.append(image_path)
        
        if len(front_faces_path) == num_samples:
            break
    return front_faces_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default="dataset/train/n000002", help='Path to the subject directory')
    parser.add_argument('--out_path', type=str, default="dataset/image_bank/front_faces.json", help='Path to the output file')
    args = parser.parse_args()
    out_path = args.out_path

    subject_name = os.path.basename(args.subject)

    verbose_dir  = "./verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)

    # Load mediapipe face landmark model
    model_asset_path = './checkpoints/models/face_landmarker_v2_with_blendshapes.task'
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)


    front_faces_path = find_front_faces(args.subject, detector=detector)

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            front_faces = json.load(f)
    else:
        front_faces = {}

    print(front_faces_path)
    with open(out_path, "w") as f:
        front_faces[subject_name] = front_faces_path
        json.dump(front_faces, f, indent=4)
        