import cv2
import pickle
import sys
import os

# =============================
# ARGUMENTS
# =============================
# sys.argv[1] = uploaded fingerprint image path
# sys.argv[2] = descriptor save path

if len(sys.argv) < 3:
    print("Usage: python enroll_fingerprint.py <image_path> <descriptor_path>")
    sys.exit(1)

image_path = sys.argv[1]
descriptor_path = sys.argv[2]

# =============================
# CHECK IMAGE EXISTS
# =============================
if not os.path.exists(image_path):
    print("Image not found")
    sys.exit(1)

# =============================
# LOAD IMAGE
# =============================
image = cv2.imread(image_path)

if image is None:
    print("Invalid image")
    sys.exit(1)

# =============================
# PREPROCESS IMAGE
# =============================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize for consistency
gray = cv2.resize(gray, (500, 500))

# Enhance contrast
gray = cv2.equalizeHist(gray)

# =============================
# ORB DETECTOR
# =============================
orb = cv2.ORB_create(nfeatures=5000)

keypoints, descriptors = orb.detectAndCompute(gray, None)

if descriptors is None:
    print("No fingerprint features detected")
    sys.exit(1)

# =============================
# SAVE DESCRIPTORS
# =============================
os.makedirs(os.path.dirname(descriptor_path), exist_ok=True)

with open(descriptor_path, "wb") as f:
    pickle.dump({
        "keypoints_count": len(keypoints),
        "descriptors": descriptors
    }, f)

print("Fingerprint enrolled successfully")