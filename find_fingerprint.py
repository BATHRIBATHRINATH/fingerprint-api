import cv2
import pickle
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# ARGUMENTS
# =============================
# sys.argv[1] = uploaded image path
# sys.argv[2] = JSON descriptor list from Node

if len(sys.argv) < 3:
    print(json.dumps({
        "error": "Invalid arguments"
    }))
    sys.exit(1)

uploaded_image_path = sys.argv[1]
with open(sys.argv[2], "r") as f:
    descriptor_list = json.load(f)

# =============================
# CHECK IMAGE
# =============================
if not os.path.exists(uploaded_image_path):
    print(json.dumps({
        "error": "Uploaded image not found"
    }))
    sys.exit(1)

# =============================
# LOAD & PREPROCESS UPLOADED IMAGE
# =============================
uploaded_img = cv2.imread(uploaded_image_path)

if uploaded_img is None:
    print(json.dumps({
        "error": "Invalid uploaded image"
    }))
    sys.exit(1)

gray_uploaded = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY)
gray_uploaded = cv2.resize(gray_uploaded, (500, 500))
gray_uploaded = cv2.equalizeHist(gray_uploaded)

# =============================
# ORB EXTRACT
# =============================
orb = cv2.ORB_create(nfeatures=5000)

kp1, des1 = orb.detectAndCompute(gray_uploaded, None)

if des1 is None:
    print(json.dumps({
        "error": "No fingerprint detected"
    }))
    sys.exit(1)

# =============================
# BF MATCHER
# =============================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

best_match = None
highest_score = 0

def match_one(fp):
    descriptor_path = fp["descriptor_path"]
    if not os.path.exists(descriptor_path):
        return None
    with open(descriptor_path, "rb") as f:
        saved_data = pickle.load(f)
    des2 = saved_data["descriptors"]
    keypoints_count = saved_data["keypoints_count"]
    if des2 is None:
        return None
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]
    max_keypoints = max(len(kp1), keypoints_count)
    if max_keypoints == 0:
        return None
    match_percentage = (len(good_matches) / max_keypoints) * 100
    return {"person_id": fp["person_id"], "blood_group": fp["blood_group"], "cloudinary_url": fp["cloudinary_url"], "match_percentage": round(match_percentage, 2)}

total = len(descriptor_list)

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(match_one, fp): i for i, fp in enumerate(descriptor_list)}
    for i, future in enumerate(as_completed(futures)):
        print(f"Checking {i+1}/{total}", file=sys.stderr)
        result = future.result()
        if result and result["match_percentage"] > highest_score:
            highest_score = result["match_percentage"]
            best_match = result
# =============================
# CONFIDENCE
# =============================
if best_match:

    score = best_match["match_percentage"]

    if score >= 95:
        confidence = "Exact Match"
    elif score >= 85:
        confidence = "Very High"
    elif score >= 70:
        confidence = "High"
    elif score >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    best_match["confidence"] = confidence

    print(json.dumps(best_match))

else:
    print(json.dumps({
        "message": "No fingerprint match found",
        "match_percentage": 0,
        "confidence": "No Match"
    }))