import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
import os

# =============================
# CLOUDINARY CONFIG
# =============================
cloudinary.config(
    cloud_name = "dl9rx32pp",
    api_key    = "577572933851461",
    api_secret = "9fJNF4E1ywOQX02kyfbPM3-iskw"
)

# =============================
# MONGODB
# =============================
MONGO_URI = "mongodb+srv://bathribathrinath02_db_user:nFfXdLPjeHMVKCnf@cluster0.5cc30a7.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["Fringerprint_blood"]
collection = db["FP_Blood"]

DATASET_FOLDER = r"D:\DOWNLOADS\BLOOD\BLOOD\DATASET"
blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

total = 0
success = 0
failed = 0

for blood_group in blood_groups:
    group_folder = os.path.join(DATASET_FOLDER, blood_group)
    if not os.path.exists(group_folder):
        continue

    for filename in os.listdir(group_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        person_id  = f"{blood_group}_{os.path.splitext(filename)[0]}"
        image_path = os.path.join(group_folder, filename)
        total += 1

        try:
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                image_path,
                folder       = f"fingerprints/{blood_group}",
                public_id    = person_id,
                overwrite    = True,
                resource_type= "image"
            )
            cloudinary_url = result["secure_url"]

            # Update MongoDB
            collection.update_one(
                {"person_id": person_id},
                {"$set": {"cloudinary_url": cloudinary_url}},
                upsert=False
            )
            success += 1
            print(f"[{success}/{total}] Uploaded {person_id} → {cloudinary_url}")

        except Exception as e:
            failed += 1
            print(f"[FAILED] {person_id} → {e}")

print(f"\nDone! Success: {success}, Failed: {failed}, Total: {total}")