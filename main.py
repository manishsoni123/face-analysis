from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import numpy as np
import cv2
from face_utils import create_repository, extract_face_encoding, store_face, list_faces, compare_faces, search_face, delete_face

BASE_DIR = "face_repositories"
os.makedirs(BASE_DIR, exist_ok=True)  # Ensure the base directory exists

app = FastAPI()

@app.get("/")
async def test():
    return {"status": "success", "message": f"API Is Working"}

@app.post("/repository/create")
async def create_repository_api(repo_name: str):
    """ Create a new repository """
    create_repository(repo_name)
    return {"status": "success", "message": f"Repository '{repo_name}' created"}

@app.post("/repository/{repo_id}/add-face")
async def add_face(repo_id: str, name: str, file: UploadFile = File(...)):
    """ Save face image and encoding in a repository """
    repo_path = os.path.join(BASE_DIR, repo_id)
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail="Repository not found")

    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_encoding = extract_face_encoding(image)
    if face_encoding is None:
        raise HTTPException(status_code=400, detail="No face detected")

    face_id = store_face(repo_id, name, face_encoding, image)

    return {"status": "success", "face_id": face_id}

@app.get("/repository/{repo_id}/list-faces")
async def list_faces_api(repo_id: str):
    """ List faces stored in a repository """
    faces = list_faces(repo_id)
    return {"status": "success", "faces": faces}

@app.post("/compare-faces")
async def compare_faces_api(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """ Compare two face images and return similarity score """
    image1_bytes = await file1.read()
    image2_bytes = await file2.read()

    image1 = cv2.imdecode(np.frombuffer(image1_bytes, np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(image2_bytes, np.uint8), cv2.IMREAD_COLOR)

    similarity = compare_faces(image1, image2)
    return {"status": "success", "similarity": similarity}

@app.post("/repository/{repo_id}/search-face")
async def search_face_api(repo_id: str, file: UploadFile = File(...)):
    """ Search a face in the repository """
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    result = search_face(repo_id, image)
    if result is None:
        return {"status": "not found"}

    return {"status": "success", "matched_face": result}

@app.delete("/repository/{repo_id}/delete-face/{face_id}")
async def delete_face_api(repo_id: str, face_id: str):
    """ Delete a stored face from the repository """
    delete_face(repo_id, face_id)
    return {"status": "success", "message": f"Face {face_id} deleted"}
