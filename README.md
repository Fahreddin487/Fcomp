# Fcomp

This is a ML app with a user interface with the following usage:

- Detect and save faces from photos,
- Create a csv file from the embeddings of faces
- Compare and sort faces for a given photo according to extracted embeddings

# Installation

- Install Python 3.9
- Install pipenv
  `pip install pipenv`
- Install requirements
  `pipenv install -r requirements.txt`
- Start streamlit
  `streamlit run UI.py --server.port = 8000`

# Usage

- Create a "Photos" folder in the working directory.
- For a given photo.jpg, save it to "./Photos/NameOfThePhoto/photo.jpg"
- Run the Streamlit
- Detect and extract the faces from the files in the ./Photos. For the photo.jpg, the result will be saved to ./Faces/NameOfThePhoto.jpg
- Extract the embeddings from the face files in the ./Faces that will be saved to ./Embedding/embeddings.csv
- Upload a photo of your desire in the UI and compare it with other photos.
