import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from numpy import expand_dims
from numpy import asarray
from PIL import Image
from numpy import asarray
import os


class AI:
    def __init__(self, *args, **kwargs):
        self.load()

    def load(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.init_files()
        self.init_embedding()

    def init_embedding(self):
        if "embeddings.csv" in os.listdir("Embedding"):
            self.embeddings = pd.read_csv("Embedding/embeddings.csv")
        else:
            self.embeddings = pd.DataFrame()
            self.embeddings.to_csv("Embedding/embeddings.csv")

    def init_files(self):
        if not "Photos" in os.listdir():
            os.mkdir("Photos")
        if not "Faces" in os.listdir():
            os.mkdir("Faces")
        if not "Embedding" in os.listdir():
            os.mkdir("Embedding")

    def extract_face(self, filename, required_size=(160, 160)):
        image = Image.open(filename)
        image = image.convert('RGB')
        pixels = asarray(image)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    def all_faces(self):
        for fileName in os.listdir("Photos"):

            file = os.listdir("Photos/" + fileName)[0]
            if not fileName + ".jpg" in os.listdir("Faces"):
                face_array = self.extract_face(
                    "Photos/" + fileName + " /" + file, required_size=(160, 160))
                face = Image.fromarray(face_array)
                face.save("Faces/" + fileName + ".jpg")

    def get_embeddings(self, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        transform = transforms.ToTensor()
        tensor = transform(face_pixels)
        tensor = tensor.unsqueeze(0)
        result = self.model(tensor)
        return result[0].detach().numpy()

    def all_embeddings(self):
        i = 0
        new_columns = dict()
        for file in os.listdir("Faces"):
            i = i+1
            print(file)
            print(f"Number of Embeddings processed = {i}")
            if file in self.embeddings.columns:
                pass
            else:
                image = Image.open("Faces" + "/" + file)
                pixels = asarray(image)
                embedding = self.get_embeddings(pixels)
                new_columns[file] = embedding
        if new_columns != {}:
            self.embeddings = pd.concat([self.embeddings, pd.DataFrame(new_columns)],
                                        ignore_index=False, axis=1)
            self.embeddings.to_csv('Embedding/embeddings.csv', index=False)

    def compare_photo(self, filename):
        facePixels = self.extract_face(
            filename, required_size=(160, 160))
        faceEmbedding = self.get_embeddings(facePixels)
        similarityIndex = dict()
        for i in self.embeddings.columns:
            similarityIndex[i] = [np.linalg.norm(
                faceEmbedding-self.embeddings[i])]
        similarityIndex = pd.DataFrame(similarityIndex)
        similarityIndex.sort_values(
            by=0, axis=1, ascending=True, inplace=True)
        return similarityIndex
