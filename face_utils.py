import os
import json
import numpy as np
import face_recognition
import uuid
import cv2

BASE_DIR = "face_repositories"

os.makedirs(BASE_DIR, exist_ok=True)  # Ensure base directory exists

def create_repository(repo_id: str):
    """ Create a new repository folder with metadata """
    repo_path = os.path.join(BASE_DIR, repo_id)
    os.makedirs(repo_path, exist_ok=True)

    # Create metadata file (if needed)
    metadata_path = os.path.join(repo_path, "metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            json.dump({"repository": repo_id, "faces": []}, f)

    return repo_path

def extract_face_encoding(image: np.ndarray):
    """ Extracts a 128D face encoding from an image """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) == 0:
        return None

    encoding = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
    return encoding[0] if encoding else None

def store_face(repo_id: str, name: str, face_encoding, image: np.ndarray):
    """ Store face encoding and image in the respective repo folder """
    repo_path = create_repository(repo_id)

    face_id = str(uuid.uuid4())

    # Save encoding to a JSON file
    encoding_path = os.path.join(repo_path, f"{face_id}.json")
    with open(encoding_path, "w") as f:
        json.dump({"name": name, "encoding": face_encoding.tolist()}, f)

    # Save image file
    image_path = os.path.join(repo_path, f"{face_id}.jpg")
    cv2.imwrite(image_path, image)

    return face_id

def list_faces(repo_id: str):
    """ List all stored faces in a repository """
    repo_path = os.path.join(BASE_DIR, repo_id)
    if not os.path.exists(repo_path):
        return []

    face_list = []
    for file in os.listdir(repo_path):
        if file.endswith(".json") and file != "metadata.json":
            with open(os.path.join(repo_path, file), "r") as f:
                face_data = json.load(f)
                face_list.append({
                    "face_id": file.replace(".json", ""),
                    "name": face_data["name"],
                    "image_path": os.path.join(repo_path, file.replace(".json", ".jpg"))
                })

    return face_list

def compare_faces(image1: np.ndarray, image2: np.ndarray):
    """ Compare two face encodings and return similarity score """
    encoding1 = extract_face_encoding(image1)
    encoding2 = extract_face_encoding(image2)

    if encoding1 is None or encoding2 is None:
        return "No face detected in one or both images"

    distance = np.linalg.norm(encoding1 - encoding2)
    similarity = 1 - (distance / np.sqrt(len(encoding1)))  # Normalize similarity score
    return round(similarity, 2)

def search_face(repo_id: str, image: np.ndarray):
    """ Search for the closest face match in a repository """
    face_encoding = extract_face_encoding(image)
    if face_encoding is None:
        return None

    repo_path = os.path.join(BASE_DIR, repo_id)
    if not os.path.exists(repo_path):
        return None

    best_match = None
    best_score = float("inf")

    for file in os.listdir(repo_path):
        if file.endswith(".json") and file != "metadata.json":
            with open(os.path.join(repo_path, file), "r") as f:
                face_data = json.load(f)
                stored_encoding = np.array(face_data["encoding"])

                distance = np.linalg.norm(stored_encoding - face_encoding)
                if distance < best_score:
                    best_score = distance
                    best_match = {"face_id": file.replace(".json", ""), "name": face_data["name"], "score": 1 - (distance / np.sqrt(len(face_encoding)))}

    return best_match if best_match else None

def delete_face(repo_id: str, face_id: str):
    """ Delete a face from the repository """
    repo_path = os.path.join(BASE_DIR, repo_id)
    json_path = os.path.join(repo_path, f"{face_id}.json")
    image_path = os.path.join(repo_path, f"{face_id}.jpg")

    if os.path.exists(json_path):
        os.remove(json_path)
    
    if os.path.exists(image_path):
        os.remove(image_path)

    return True
