from flask import Flask, request, jsonify
import dlib
import numpy as np

app = Flask(_name_)

# Dlib face detector and recognition model
detector = dlib.get_frontal_face_detector()
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

@app.route('/encode', methods=['POST'])
def encode_face():
    data = request.json
    image_url = data.get('imageUrl')

    # Load image from URL and process
    img = dlib.load_rgb_image(image_url)
    faces = detector(img, 1)

    if len(faces) == 1:
        shape = predictor(img, faces[0])
        face_encoding = np.array(face_rec_model.compute_face_descriptor(img, shape), dtype=np.float32)

        return jsonify({"face_encoding": face_encoding.tolist()})
    else:
        return jsonify({"error": "Face encoding failed. Ensure exactly one face is in the photo."}), 400

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    data = request.json
    image_url = data.get('imageUrl')

    # Load image from URL and process
    img = dlib.load_rgb_image(image_url)
    faces = detector(img, 1)

    recognized_names = []
    for face in faces:
        shape = predictor(img, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(img, shape), dtype=np.float32)

        # Compare face encoding with the database to find matches (Insert matching logic here)
        recognized_names.append("Matched Name")

    return jsonify({"recognized_names": recognized_names})

if _name_ == '_main_':
    app.run(port=5000, debug=True)