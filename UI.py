import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from main import AI
import os


class UI(AI):

    def __init__(self):
        super().__init__(self)

    @st.cache
    def compare_photo(self, filename):
        df = super().compare_photo(filename)
        return df

    def all_embeddings(self):
        my_bar = st.progress(0)
        currentFile = st.empty()
        i = 0
        new_columns = dict()

        for file in os.listdir("Faces"):
            currentFile.write(file)
            i = i+1
            if file in self.embeddings.columns:
                pass
            else:
                image = Image.open("Faces" + "/" + file)
                pixels = np.asarray(image)
                embedding = self.get_embeddings(pixels)
                new_columns[file] = embedding
            my_bar.progress(i/len(os.listdir("Faces")))

        if new_columns != {}:
            self.embeddings = pd.concat([self.embeddings, pd.DataFrame(new_columns)],
                                        ignore_index=False, axis=1)
            self.embeddings.to_csv('Embedding/embeddings.csv', index=False)
        my_bar.empty()
        currentFile.empty()
        st.success("Success!!")

    def all_faces(self):
        currentFile = st.empty()
        my_bar = st.progress(0)
        i = 0
        for fileName in os.listdir("Photos"):
            currentFile.write(fileName)
            i = i+1
            if not (fileName + ".jpg" in os.listdir("Faces")):
                try:
                    file = os.listdir("Photos/" + fileName)[0]
                    face_array = self.extract_face(
                        "Photos/" + fileName + "/" + file, required_size=(160, 160))

                    face = Image.fromarray(face_array)

                    face.save("Faces/" + fileName + ".jpg")
                except:
                    pass
            my_bar.progress(i/len(os.listdir("Photos")))
        my_bar.empty()
        currentFile.empty()
        st.success("Success!!")


ui = UI()

st.title('Face comparison App')

st.sidebar.markdown("Select")

imgFile = st.sidebar.file_uploader('Upload Image')

if imgFile != None:
    try:
        if "fileName" not in st.session_state:
            st.session_state.fileName = imgFile.name

        if st.session_state.fileName != imgFile.name or 'showsPhotos' not in st.session_state:
            st.session_state.showsPhotos = False

        st.image(imgFile, caption='Uploaded Image.', width=500)
        select_event = st.sidebar.selectbox("What do you want to do with the image?", [
            "Find Similar Faces", "Add to 'Photos'"])

        if select_event == "Find Similar Faces":

            numPhotoShown = st.sidebar.number_input(
                'Select number of photos to shown', 1, 20)

            similarityTitle = st.empty()
            similarity = st.empty()

            if st.sidebar.button("Find") or st.session_state.showsPhotos:
                st.session_state.showsPhotos = True
                similarityIndex = ui.compare_photo(imgFile)
                similarity.dataframe(similarityIndex)
                st.write("Similar images")
                columns = st.columns(4)
                for i in range(numPhotoShown):
                    fileName = similarityIndex.columns[i].split(".")[0]

                    try:
                        columns[i % 4].image("Photos/" + fileName + "/" + os.listdir("Photos/" + fileName)[0],
                                             caption=f'Similar Image {i+1}: {fileName}')
                    except:
                        pass

                similarityTitle.markdown("Similarity index")
    except:
        st.sidebar.markdown("Input Error")

# st.markdown(" All Embeddings")
place_holder = st.empty()
place_holder.dataframe(ui.embeddings)

if 'showsPhotos' not in st.session_state:
    st.session_state.showsPhotos = False


st.sidebar.header(
    f" {len(os.listdir('Faces'))}/{len(os.listdir('Photos'))} : Faces/Photos")
st.sidebar.header(
    f"{len(ui.embeddings.columns)}/{len(os.listdir('Faces'))} : Embeddings/Faces")


option = st.sidebar.selectbox('Select', ["Extract Faces from 'Photos'",
                                         "Extract Embeddings from 'Faces'"])

if st.sidebar.button("Confirm", key=2):

    if option == "Extract Faces from 'Photos'":
        ui.all_faces()
    elif option == "Extract Embeddings from 'Faces'":
        ui.all_embeddings()
