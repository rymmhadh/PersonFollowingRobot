# RobotDB.py

import pymongo
import numpy as np

# Connexion MongoDB (à adapter selon ta config)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["robotDB"]
collection = db["profiles"]

def tensor_to_list(tensor):
    if tensor is None:
        return None
    # Si c'est un tenseur PyTorch
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().tolist()
    # Si c'est déjà une liste ou un numpy array
    elif isinstance(tensor, (list, np.ndarray)):
        return tensor
    else:
        raise TypeError(f"Unsupported type for tensor_to_list: {type(tensor)}")

def save_profile_to_db(name, face_emb, body_front_emb, body_rear_emb):
    # Préparation du document à insérer
    profile_doc = {
        "name": name,
        "face_embedding": tensor_to_list(face_emb),
        "front_body_embedding": tensor_to_list(body_front_emb),
        "rear_body_embedding": tensor_to_list(body_rear_emb)
    }

    # Insertion dans la collection MongoDB
    result = collection.insert_one(profile_doc)
    print(f"[INFO] Profile saved with id: {result.inserted_id}")
