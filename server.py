from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pickle
import subprocess
import shutil
import uuid
import json
import sys
import os
import cloudinary
import cloudinary.uploader


cloudinary.config(
    cloud_name = "dl9rx32pp",
    api_key    = "577572933851461",
    api_secret = "9fJNF4E1ywOQX02kyfbPM3-iskw"
)

# =============================
# APP INIT
# =============================
app = FastAPI(title="Fingerprint Matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# MONGODB CONNECTION
# =============================
MONGO_URI = "mongodb+srv://bathribathrinath02_db_user:nFfXdLPjeHMVKCnf@cluster0.5cc30a7.mongodb.net/?appName=Cluster0"
DB_NAME = "Fringerprint_blood"
COLLECTION_NAME = "FP_Blood"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# =============================
# PATHS CONFIG
# =============================
# Folder where uploaded fingerprint images are saved temporarily
UPLOAD_FOLDER = "./uploads"

# Folder where .pkl descriptor files are saved
DESCRIPTOR_FOLDER = "./descriptors"

# Dataset folder (blood group wise images already stored)
DATASET_FOLDER = r"D:/DOWNLOADS/BLOOD/BLOOD/DATASET"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DESCRIPTOR_FOLDER, exist_ok=True)

print("Loading all descriptors into RAM...")
descriptor_cache = []
for record in collection.find({}, {"_id": 0}):
    path = record.get("descriptor_path", "")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            saved_data = pickle.load(f)
        descriptor_cache.append({
            "person_id":       record.get("person_id", ""),
            "blood_group":     record.get("blood_group", ""),
            "cloudinary_url":  record.get("cloudinary_url", ""),
            "descriptors":     saved_data["descriptors"],
            "keypoints_count": saved_data["keypoints_count"]
        })
print(f"Loaded {len(descriptor_cache)} descriptors into RAM ✅")

# =============================
# HELPER - Run Python Script
# =============================
def run_python_script(script_name, args):
    result = subprocess.run(
        [sys.executable, script_name] + args,
        capture_output=True,
        text=True
    )
    return result.stdout.strip(), result.stderr.strip()


# =============================
# ROOT
# =============================
@app.get("/")
def root():
    return {"message": "Fingerprint API is running"}


# =============================
# ROUTE 1: ENROLL FINGERPRINT
# POST /api/enroll
# =============================
@app.post("/api/enroll")
async def enroll_fingerprint(
    image: UploadFile = File(...),
    person_id: str = Form(...),
    blood_group: str = Form(...),
    name: str = Form(None),
):
    # Save uploaded image temporarily
    temp_image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{image.filename}")

    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Descriptor save path
    descriptor_path = os.path.join(DESCRIPTOR_FOLDER, f"{person_id}.pkl")
    descriptor_path = os.path.abspath(descriptor_path)

    try:
        # Run enroll_fingerprint.py
        stdout, stderr = run_python_script("enroll_fingerprint.py", [temp_image_path, descriptor_path])

        if "successfully" not in stdout.lower():
            raise HTTPException(status_code=400, detail=f"Enroll failed: {stdout or stderr}")

        # Check if person already exists in DB
        existing = collection.find_one({"person_id": person_id})

        if existing:
            # Update existing record
            collection.update_one(
                {"person_id": person_id},
                {"$set": {
                    "blood_group": blood_group,
                    "name": name,
                    "descriptor_path": descriptor_path,
                    "image_filename": image.filename
                }}
            )
            message = "Fingerprint updated successfully"
        else:
            # Insert new record
            collection.insert_one({
                "person_id": person_id,
                "name": name,
                "blood_group": blood_group,
                "descriptor_path": descriptor_path,
                "image_filename": image.filename,
                "cloudinary_url": ""
            })
            message = "Fingerprint enrolled successfully"

        return JSONResponse(content={
            "success": True,
            "message": message,
            "person_id": person_id,
            "blood_group": blood_group,
            "descriptor_path": descriptor_path
        })

    finally:
        # Cleanup temp image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# =============================
# ROUTE 2: MATCH FINGERPRINT
# POST /api/match
# =============================
@app.post("/api/match")
async def match_fingerprint(
    image: UploadFile = File(...)
):
    # Save uploaded image temporarily
    temp_image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{image.filename}")
    temp_json_path  = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_descriptors.json")

    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        # Fetch all enrolled fingerprints from MongoDB
        all_records = list(collection.find({}, {"_id": 0}))

        if not all_records:
            return JSONResponse(content={
                "success": False,
                "message": "No fingerprints enrolled in database"
            })

        # Build descriptor list for find_fingerprint.py
        descriptor_list = []
        for record in all_records:
            pkl_bytes = record.get("pkl_data")
            if pkl_bytes:
                # Write pkl_data from MongoDB to temp file
                temp_pkl = os.path.join(UPLOAD_FOLDER, f"{record.get('person_id')}.pkl")
                with open(temp_pkl, "wb") as f:
                    f.write(pkl_bytes)
                descriptor_list.append({
                    "person_id": record.get("person_id", ""),
                    "blood_group": record.get("blood_group", ""),
                    "cloudinary_url": record.get("cloudinary_url", ""),
                    "descriptor_path": temp_pkl
                })

        if not descriptor_list:
            return JSONResponse(content={
                "success": False,
                "message": "No valid descriptor files found on server"
            })

        # Run find_fingerprint.py
        with open(temp_json_path, "w") as f:
            json.dump(descriptor_list, f)

        stdout, stderr = run_python_script(
            "find_fingerprint.py",
            [temp_image_path, temp_json_path]
)

        if not stdout:
            raise HTTPException(status_code=500, detail=f"Script error: {stderr}")

        result = json.loads(stdout)

        return JSONResponse(content={
            "success": True,
            **result
        })

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid script output: {stdout}")

    finally:
        # Cleanup temp image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)


# =============================
# ROUTE 3: BULK ENROLL FROM DATASET FOLDER
# POST /api/enroll-dataset
# This reads D:/DOWNLOADS/BLOOD/BLOOD/DATASET
# and enrolls all images blood group wise
# =============================
@app.post("/api/enroll-dataset")
async def enroll_dataset():

    if not os.path.exists(DATASET_FOLDER):
        raise HTTPException(status_code=404, detail=f"Dataset folder not found: {DATASET_FOLDER}")

    enrolled = []
    failed = []

    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

    for blood_group in blood_groups:
        group_folder = os.path.join(DATASET_FOLDER, blood_group)

        if not os.path.exists(group_folder):
            continue

        for filename in list(os.listdir(group_folder))[:10]:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            image_path = os.path.join(group_folder, filename)
            person_id = f"{blood_group}_{os.path.splitext(filename)[0]}"
            descriptor_path = os.path.abspath(os.path.join(DESCRIPTOR_FOLDER, f"{person_id}.pkl"))

            # Run enroll script
            stdout, stderr = run_python_script("enroll_fingerprint.py", [image_path, descriptor_path])

            if "successfully" in stdout.lower():
                # Upload to Cloudinary
                upload_result = cloudinary.uploader.upload(
                    image_path,
                    folder="fingerprints",
                    public_id=person_id
                )
                cloudinary_url = upload_result.get("secure_url", "")

                # Read .pkl file as binary
                with open(descriptor_path, "rb") as f:
                    pkl_data = f.read()
    
                # Upsert into MongoDB with Cloudinary URL
                collection.update_one(
                    {"person_id": person_id},
                    {"$set": {
                        "person_id": person_id,
                        "blood_group": blood_group,
                        "descriptor_path": descriptor_path,
                        "image_filename": filename,
                        "cloudinary_url": cloudinary_url,
                        "pkl_data": pkl_data
                    }},
                    upsert=True
                )
                enrolled.append(person_id)
            else:
                failed.append({"person_id": person_id, "error": stdout or stderr})

    return JSONResponse(content={
        "success": True,
        "enrolled_count": len(enrolled),
        "failed_count": len(failed),
        "enrolled": enrolled,
        "failed": failed
    })


# =============================
# ROUTE 4: LIST ALL ENROLLED
# GET /api/persons
# =============================
@app.get("/api/persons")
def list_persons():
    records = list(collection.find({}, {"_id": 0, "descriptor_path": 0}))
    return JSONResponse(content={
        "success": True,
        "count": len(records),
        "persons": records
    })


# =============================
# ROUTE 5: DELETE A PERSON
# DELETE /api/person/{person_id}
# =============================
@app.delete("/api/person/{person_id}")
def delete_person(person_id: str):
    record = collection.find_one({"person_id": person_id})

    if not record:
        raise HTTPException(status_code=404, detail="Person not found")

    # Delete .pkl file
    descriptor_path = record.get("descriptor_path", "")
    if descriptor_path and os.path.exists(descriptor_path):
        os.remove(descriptor_path)

    # Delete from MongoDB
    collection.delete_one({"person_id": person_id})

    return JSONResponse(content={
        "success": True,
        "message": f"Person {person_id} deleted successfully"
    })
