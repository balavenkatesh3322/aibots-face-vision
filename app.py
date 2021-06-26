import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple
import time

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


import math
import os


from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions
from deepface.detectors import OpenCvWrapper
from tqdm import tqdm
import pandas as pd
import re



HERE = Path(__file__).parent

logger = logging.getLogger(__name__)




# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)



def aibots():

    class Detection(NamedTuple):
        name: str
        prob: float

    class OpenCVVideoProcessor(VideoProcessorBase):

        #confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self.emotion_model = DeepFace.build_model('Emotion')
            print("MAin Emotion model loaded")

            self.age_model = DeepFace.build_model('Age')
            print("MAin Age model loaded")

            self.gender_model = DeepFace.build_model('Gender')
            print("MAin Gender model loaded")
            #self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            db_path = ''
            model_name ='VGG-Face'
            distance_metric = 'cosine'
            enable_face_analysis = True
            source = 0
            time_threshold = 1
            frame_threshold =1


            input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

            text_color = (255,255,255)

            # employees = []
            # #check passed db folder exists
            # if os.path.isdir(db_path) == True:
            #     for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
            #         for file in f:
            #             if ('.jpg' in file):
            #                 #exact_path = os.path.join(r, file)
            #                 exact_path = r + "/" + file
            #                 #print(exact_path)
            #                 employees.append(exact_path)

            # if len(employees) == 0:
            #     print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")

            # #------------------------

            # if len(employees) > 0:

            #     model = DeepFace.build_model(model_name)
            #     print(model_name," is built")

            #     #------------------------

            #     input_shape = functions.find_input_shape(model)
            #     input_shape_x = input_shape[0]
            #     input_shape_y = input_shape[1]

            #     #tuned thresholds for model and metric pair
            #     threshold = dst.findThreshold(model_name, distance_metric)

            #------------------------
            #facial attribute analysis models

            # if enable_face_analysis == True:

            #     tic = time.time()

            #     emotion_model = DeepFace.build_model('Emotion')
            #     print("Emotion model loaded")

            #     age_model = DeepFace.build_model('Age')
            #     print("Age model loaded")

            #     gender_model = DeepFace.build_model('Gender')
            #     print("Gender model loaded")

            #     toc = time.time()

            #     print("Facial attibute analysis models loaded in ",toc-tic," seconds")

            #------------------------

            #find embeddings for employee list

            #tic = time.time()

            # pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

            # embeddings = []
            # #for employee in employees:
            # for index in pbar:
            #     employee = employees[index]
            #     pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
            #     embedding = []
            #     img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False)
            #     img_representation = model.predict(img)[0,:]

            #     embedding.append(employee)
            #     embedding.append(img_representation)
            #     embeddings.append(embedding)

            # df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
            # df['distance_metric'] = distance_metric

            # toc = time.time()

            # print("Embeddings found for given data set in ", toc-tic," seconds")

            #-----------------------

            pivot_img_size = 112 #face recognition result image

            #-----------------------

            opencv_path = OpenCvWrapper.get_opencv_path()
            face_detector_path = opencv_path+"haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(face_detector_path)

            #-----------------------

            freeze = False
            face_detected = False
            face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
            freezed_frame = 0
            tic = time.time()

            #result: List[Detection] = []

            result = []

            #cap = cv2.VideoCapture(source) #webcam

            #while(True):
            #ret, img = cap.read()

           #if img is None:
               # break

            #cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
            #cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            raw_img = img.copy()
            resolution = img.shape

            resolution_x = img.shape[1]; resolution_y = img.shape[0]

            if freeze == False:
                faces = face_cascade.detectMultiScale(img, 1.3, 5)

                if len(faces) == 0:
                    face_included_frames = 0
            else:
                faces = []

            detected_faces = []
            face_index = 0
            for (x,y,w,h) in faces:
                if w > 130: #discard small detected faces

                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1 #increase frame for a single face

                    cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                    #cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

                    #-------------------------------------

                    detected_faces.append((x,y,w,h))
                    face_index = face_index + 1

                    #-------------------------------------

            if face_detected == True and face_included_frames == frame_threshold and freeze == False:
                freeze = True
                #base_img = img.copy()
                base_img = raw_img.copy()
                detected_faces_final = detected_faces.copy()
                tic = time.time()

            if freeze == True:

                toc = time.time()
                if (toc - tic) < time_threshold:

                    if freezed_frame == 0:
                        freeze_img = base_img.copy()
                        #freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

                        for detected_face in detected_faces_final:
                            x = detected_face[0]; y = detected_face[1]
                            w = detected_face[2]; h = detected_face[3]

                            #cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                            #-------------------------------

                            #apply deep learning for custom_face

                            custom_face = base_img[y:y+h, x:x+w]

                            #-------------------------------
                            #facial attribute analysis

                            if enable_face_analysis == True:

                                gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = False)
                                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                                emotion_predictions = self.emotion_model.predict(gray_img)[0,:]
                                sum_of_predictions = emotion_predictions.sum()


                                mood_items = []
                                for i in range(0, len(emotion_labels)):
                                    mood_item = []
                                    emotion_label = emotion_labels[i]
                                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                                    mood_item.append(emotion_label)
                                    mood_item.append(emotion_prediction)
                                    mood_items.append(mood_item)

                                emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
                                emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

                                #result.append([emotion_name=emotion_label, prob=float(emotion_prediction)])
                                result.append(emotion_label)

                                #background of mood box

                                #transparency
                                overlay = freeze_img.copy()
                                opacity = 0.4

                                # if x+w+pivot_img_size < resolution_x:
                                #     #right
                                #     cv2.rectangle(freeze_img
                                #         #, (x+w,y+20)
                                #         , (x+w,y)
                                #         , (x+w+pivot_img_size, y+h)
                                #         , (64,64,64),cv2.FILLED)

                                #     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                # elif x-pivot_img_size > 0:
                                #     #left
                                #     cv2.rectangle(freeze_img
                                #         #, (x-pivot_img_size,y+20)
                                #         , (x-pivot_img_size,y)
                                #         , (x, y+h)
                                #         , (64,64,64),cv2.FILLED)

                                #     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                                # for index, instance in emotion_df.iterrows():
                                #     emotion_label = "%s " % (instance['emotion'])
                                #     emotion_score = instance['score']/100

                                #     bar_x = 35 #this is the size if an emotion is 100%
                                #     bar_x = int(bar_x * emotion_score)

                                #     if x+w+pivot_img_size < resolution_x:

                                #         text_location_y = y + 20 + (index+1) * 20
                                #         text_location_x = x+w

                                #         if text_location_y < y + h:
                                #             #cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                #             cv2.rectangle(freeze_img
                                #                 , (x+w+70, y + 13 + (index+1) * 20)
                                #                 , (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                #                 , (255,255,255), cv2.FILLED)

                                #     elif x-pivot_img_size > 0:

                                #         text_location_y = y + 20 + (index+1) * 20
                                #         text_location_x = x-pivot_img_size

                                #         if text_location_y <= y+h:
                                #             #cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                #             cv2.rectangle(freeze_img
                                #                 , (x-pivot_img_size+70, y + 13 + (index+1) * 20)
                                #                 , (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                #                 , (255,255,255), cv2.FILLED)

                                #-------------------------------

                                face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False)

                                age_predictions = self.age_model.predict(face_224)[0,:]
                                apparent_age = Age.findApparentAge(age_predictions)

                                sum_of_predictions = age_predictions.sum()
                                age_prop = 100 * age_predictions[i] / sum_of_predictions
                                #result.append([Age=int(apparent_age), prob=float(age_prop)])
                                result.append(int(apparent_age))

                                #-------------------------------

                                gender_prediction = self.gender_model.predict(face_224)[0,:]


                                if np.argmax(gender_prediction) == 0:
                                    gender = "Female"
                                elif np.argmax(gender_prediction) == 1:
                                    gender = "Male"

                                #print(str(int(apparent_age))," years old ", dominant_emotion, " ", gender)

                                analysis_report = str(int(apparent_age))+" "+gender

                                #sum_of_predictions = gender_prediction.sum()
                                #gender_prop = 100 * gender_prediction[i] / sum_of_predictions
                                result.append(str(gender))

                                #-------------------------------

                                info_box_color = (46,200,255)

                                # #top
                                # if y - pivot_img_size + int(pivot_img_size/5) > 0:

                                #     triangle_coordinates = np.array( [
                                #         (x+int(w/2), y)
                                #         , (x+int(w/2)-int(w/10), y-int(pivot_img_size/3))
                                #         , (x+int(w/2)+int(w/10), y-int(pivot_img_size/3))
                                #     ] )

                                #     cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

                                #     cv2.rectangle(freeze_img, (x+int(w/5), y-pivot_img_size+int(pivot_img_size/5)), (x+w-int(w/5), y-int(pivot_img_size/3)), info_box_color, cv2.FILLED)

                                #     cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y - int(pivot_img_size/2.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

                                # #bottom
                                # elif y + h + pivot_img_size - int(pivot_img_size/5) < resolution_y:

                                #     triangle_coordinates = np.array( [
                                #         (x+int(w/2), y+h)
                                #         , (x+int(w/2)-int(w/10), y+h+int(pivot_img_size/3))
                                #         , (x+int(w/2)+int(w/10), y+h+int(pivot_img_size/3))
                                #     ] )

                                #     cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

                                #     cv2.rectangle(freeze_img, (x+int(w/5), y + h + int(pivot_img_size/3)), (x+w-int(w/5), y+h+pivot_img_size-int(pivot_img_size/5)), info_box_color, cv2.FILLED)

                                #     cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y + h + int(pivot_img_size/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

                            #-------------------------------
                            #face recognition

                            # custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = False)

                            # #check preprocess_face function handled
                            # if custom_face.shape[1:3] == input_shape:
                            #     if df.shape[0] > 0: #if there are images to verify, apply face recognition
                            #         img1_representation = model.predict(custom_face)[0,:]

                            #         #print(freezed_frame," - ",img1_representation[0:5])

                            #         def findDistance(row):
                            #             distance_metric = row['distance_metric']
                            #             img2_representation = row['embedding']

                            #             distance = 1000 #initialize very large value
                            #             if distance_metric == 'cosine':
                            #                 distance = dst.findCosineDistance(img1_representation, img2_representation)
                            #             elif distance_metric == 'euclidean':
                            #                 distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                            #             elif distance_metric == 'euclidean_l2':
                            #                 distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

                            #             return distance

                            #         df['distance'] = df.apply(findDistance, axis = 1)
                            #         df = df.sort_values(by = ["distance"])

                            #         candidate = df.iloc[0]
                            #         employee_name = candidate['employee']
                            #         best_distance = candidate['distance']

                            #         #print(candidate[['employee', 'distance']].values)

                            #         #if True:
                            #         if best_distance <= threshold:
                            #             #print(employee_name)
                            #             display_img = cv2.imread(employee_name)

                            #             display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

                            #             label = employee_name.split("/")[-1].replace(".jpg", "")
                            #             label = re.sub('[0-9]', '', label)

                            #             try:
                            #                 if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                            #                     #top right
                            #                     freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img

                            #                     overlay = freeze_img.copy(); opacity = 0.4
                            #                     cv2.rectangle(freeze_img,(x+w,y),(x+w+pivot_img_size, y+20),(46,200,255),cv2.FILLED)
                            #                     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                            #                     cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                            #                     #connect face and text
                            #                     cv2.line(freeze_img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                            #                     cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)

                            #                 elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                            #                     #bottom left
                            #                     freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img

                            #                     overlay = freeze_img.copy(); opacity = 0.4
                            #                     cv2.rectangle(freeze_img,(x-pivot_img_size,y+h-20),(x, y+h),(46,200,255),cv2.FILLED)
                            #                     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                            #                     cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                            #                     #connect face and text
                            #                     cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                            #                     cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)

                            #                 elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                            #                     #top left
                            #                     freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img

                            #                     overlay = freeze_img.copy(); opacity = 0.4
                            #                     cv2.rectangle(freeze_img,(x- pivot_img_size,y),(x, y+20),(46,200,255),cv2.FILLED)
                            #                     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                            #                     cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                            #                     #connect face and text
                            #                     cv2.line(freeze_img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                            #                     cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)

                            #                 elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                            #                     #bottom righ
                            #                     freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img

                            #                     overlay = freeze_img.copy(); opacity = 0.4
                            #                     cv2.rectangle(freeze_img,(x+w,y+h-20),(x+w+pivot_img_size, y+h),(46,200,255),cv2.FILLED)
                            #                     cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

                            #                     cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                            #                     #connect face and text
                            #                     cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                            #                     cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
                            #             except Exception as err:
                            #                 print(str(err))

                            # tic = time.time() #in this way, freezed image can show 5 seconds

                            #-------------------------------

                    time_left = int(time_threshold - (toc - tic) + 1)

                    #cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
                    #cv2.putText(freeze_img, str(time_left), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    #cv2.imshow('img', freeze_img)

                    freezed_frame = freezed_frame + 1
                else:
                    face_detected = False
                    face_included_frames = 0
                    freeze = False
                    freezed_frame = 0

            else:
                #cv2.imshow('img',img)
                freeze_img = img

            self.result_queue.put(result)


            return av.VideoFrame.from_ndarray(freeze_img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="aibots-demo",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )


    if st.checkbox("Show the detected labels", value=True):
        persist_result = []
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    if result and result is not None:
                        persist_result.append(result)
                        labels_placeholder.table(persist_result)
                    # else:
                    #     labels_placeholder.table(result)
                else:
                    break


def main():
    st.header("AIBOTS DEMO")

    aibots_demo = "Age Gender Emotion Detection"

    #object_detection_page = "Real time object detection"
    video_filters_page = (
        "Face filters"
    )
    #audio_filter_page = "Real time audio filter (sendrecv)"
    #delayed_echo_page = "Delayed echo (sendrecv)"
    # streaming_page = (
    #     "Consuming media files on server-side and streaming it to browser (recvonly)"
    # )
    # video_sendonly_page = (
    #     "WebRTC is sendonly and images are shown via st.image() (sendonly)"
    # )
    # audio_sendonly_page = (
    #     "WebRTC is sendonly and audio frames are visualized with matplotlib (sendonly)"
    # )
    #loopback_page = "Simple video and audio loopback (sendrecv)"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            aibots_demo,
            #object_detection_page,
            video_filters_page,
            #audio_filter_page,
            #delayed_echo_page,
            #streaming_page,
            #video_sendonly_page,
            #audio_sendonly_page,
            #loopback_page,
        ],
    )
    st.subheader(app_mode)


    if app_mode == video_filters_page:
        app_video_filters()
    # elif app_mode == object_detection_page:
    #     app_object_detection()
    elif app_mode == aibots_demo:
        aibots()

    # elif app_mode == audio_filter_page:
    #     app_audio_filter()
    # elif app_mode == delayed_echo_page:
    #     app_delayed_echo()
    #elif app_mode == streaming_page:
        #app_streaming()
    # elif app_mode == video_sendonly_page:
    #     app_sendonly_video()
    # elif app_mode == audio_sendonly_page:
    #     app_sendonly_audio()
    # elif app_mode == loopback_page:
    #     app_loopback()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_loopback():
    """ Simple video loopback """
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=None,  # NoOp
    )


def app_video_filters():
    """ Video transforms with OpenCV """

    class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]

        def __init__(self) -> None:
            self.type = "noop"

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )


def app_audio_filter():
    DEFAULT_GAIN = 1.0

    class AudioProcessor(AudioProcessorBase):
        gain = DEFAULT_GAIN

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            raw_samples = frame.to_ndarray()
            sound = pydub.AudioSegment(
                data=raw_samples.tobytes(),
                sample_width=frame.format.bytes,
                frame_rate=frame.sample_rate,
                channels=len(frame.layout.channels),
            )

            sound = sound.apply_gain(self.gain)

            # Ref: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples  # noqa
            channel_sounds = sound.split_to_mono()
            channel_samples = [s.get_array_of_samples() for s in channel_sounds]
            new_samples: np.ndarray = np.array(channel_samples).T
            new_samples = new_samples.reshape(raw_samples.shape)

            new_frame = av.AudioFrame.from_ndarray(
                new_samples, layout=frame.layout.name
            )
            new_frame.sample_rate = frame.sample_rate
            return new_frame

    webrtc_ctx = webrtc_streamer(
        key="audio-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx.audio_processor:
        webrtc_ctx.audio_processor.gain = st.slider(
            "Gain", -10.0, +20.0, DEFAULT_GAIN, 0.05
        )


def app_delayed_echo():
    DEFAULT_DELAY = 1.0

    class VideoProcessor(VideoProcessorBase):
        delay = DEFAULT_DELAY

        async def recv_queued(self, frames: List[av.VideoFrame]) -> List[av.VideoFrame]:
            logger.debug("Delay:", self.delay)
            await asyncio.sleep(self.delay)
            return frames

    class AudioProcessor(AudioProcessorBase):
        delay = DEFAULT_DELAY

        async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
            await asyncio.sleep(self.delay)
            return frames

    webrtc_ctx = webrtc_streamer(
        key="delay",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=VideoProcessor,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor and webrtc_ctx.audio_processor:
        delay = st.slider("Delay", 0.0, 5.0, DEFAULT_DELAY, 0.05)
        webrtc_ctx.video_processor.delay = delay
        webrtc_ctx.audio_processor.delay = delay


def app_object_detection():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=MobileNetSSDVideoProcessor,
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break


def app_streaming():
    """ Media streamings """
    MEDIAFILES = {
        "big_buck_bunny_720p_2mb.mp4": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_2mb.mp4",
            "type": "video",
        },
        "big_buck_bunny_720p_10mb.mp4": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
            "type": "video",
        },
        "file_example_MP3_700KB.mp3": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
            "type": "audio",
        },
        "file_example_MP3_5MG.mp3": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
            "type": "audio",
        },
    }
    media_file_label = st.radio(
        "Select a media file to stream", tuple(MEDIAFILES.keys())
    )
    media_file_info = MEDIAFILES[media_file_label]
    download_file(media_file_info["url"], media_file_info["local_file_path"])

    def create_player():
        return MediaPlayer(str(media_file_info["local_file_path"]))

        # NOTE: To stream the video from webcam, use the code below.
        # return MediaPlayer(
        #     "1:none",
        #     format="avfoundation",
        #     options={"framerate": "30", "video_size": "1280x720"},
        # )

    WEBRTC_CLIENT_SETTINGS.update(
        {
            "media_stream_constraints": {
                "video": media_file_info["type"] == "video",
                "audio": media_file_info["type"] == "audio",
            }
        }
    )

    webrtc_streamer(
        key=f"media-streaming-{media_file_label}",
        mode=WebRtcMode.RECVONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        player_factory=create_player,
    )


def app_sendonly_video():
    """A sample to use WebRTC in sendonly mode to transfer frames
    from the browser to the server and to render frames via `st.image`."""
    webrtc_ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )

    image_place = st.empty()

    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            img_rgb = video_frame.to_ndarray(format="rgb24")
            image_place.image(img_rgb)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


def app_sendonly_audio():
    """A sample to use WebRTC in sendonly mode to transfer audio frames
    from the browser to the server and visualize them with matplotlib
    and `st.pyplog`."""
    webrtc_ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )

    fig_place = st.empty()

    fig, [ax_time, ax_freq] = plt.subplots(
        2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2}
    )

    sound_window_len = 5000  # 5s
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/  # noqa
                sound_window_buffer = sound_window_buffer.set_channels(
                    1
                )  # Stereo to mono
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_xlabel("Time")
                ax_time.set_ylabel("Magnitude")

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(sample.shape[0], 1.0 / sound_chunk.frame_rate)
                freq = freq[: int(freq.shape[0] / 2)]
                spec = spec[: int(spec.shape[0] / 2)]
                spec[0] = spec[0] / 2

                ax_freq.cla()
                ax_freq.plot(freq, np.abs(spec))
                ax_freq.set_xlabel("Frequency")
                ax_freq.set_yscale("log")
                ax_freq.set_ylabel("Magnitude")

                fig_place.pyplot(fig)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)


    main()
