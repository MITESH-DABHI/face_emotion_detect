import cv2
import threading
from deepface import DeepFace
import pandas as pd

import pyaudio
import wave
import numpy as np
import mediapipe as mp

emotion_lst = []


import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os


# class VideoRecorder:

#     # Video class based on openCV
#     def __init__(self):

#         self.open = True
#         self.device_index = 0
#         self.fps = 6  # fps should be the minimum constant rate at which the camera can
#         self.fourcc = "MJPG"  # capture images (with no decrease in speed over time; testing is required)
#         self.frameSize = (
#             640,
#             480,
#         )  # video formats and sizes also depend and vary according to the camera used
#         self.video_filename = "temp_video.avi"
#         self.video_cap = cv2.VideoCapture(self.device_index)

#         self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
#         self.video_out = cv2.VideoWriter(
#             self.video_filename, self.video_writer, self.fps, self.frameSize
#         )
#         self.frame_counts = 1
#         self.start_time = time.time()

#     # Video starts being recorded
#     def record(self):

#         # 		counter = 1
#         timer_start = time.time()
#         timer_current = 0

#         while self.open == True:
#             ret, video_frame = self.video_cap.read()
#             if ret == True:

#                 self.video_out.write(video_frame)
#                 # 					print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
#                 self.frame_counts += 1
#                 # 					counter += 1
#                 # 					timer_current = time.time() - timer_start
#                 time.sleep(0.16)

#                 # Uncomment the following three lines to make the video to be
#                 # displayed to screen while recording

#             # 					gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
#             # 					cv2.imshow('video_frame', gray)
#             # 					cv2.waitKey(1)
#             else:
#                 break

#                 # 0.16 delay -> 6 fps
#                 #

#     # Finishes the video recording therefore the thread too
#     def stop(self):

#         if self.open == True:

#             self.open = False
#             self.video_out.release()
#             self.video_cap.release()
#             cv2.destroyAllWindows()

#         else:
#             pass

#     # Launches the video recording function using a thread
#     def start(self):
#         video_thread = threading.Thread(target=self.record)
#         video_thread.start()


class AudioRecorder:

    # Audio class based on pyAudio and Wave
    def __init__(self):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        self.audio_frames = []

    # Audio starts being recorded
    def record(self):

        self.stream.start_stream()
        while self.open == True:
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if self.open == False:
                break

    # Finishes the audio recording therefore the thread too
    def stop(self):

        if self.open == True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, "wb")
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b"".join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()


def start_AVrecording(filename):

    # global video_thread
    global audio_thread

    # video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread.start()
    # video_thread.start()

    return filename


def start_video_recording(filename):

    global video_thread

    # video_thread = VideoRecorder()
    # video_thread.start()

    return filename


def start_audio_recording(filename):

    global audio_thread

    audio_thread = AudioRecorder()
    audio_thread.start()

    return filename


def stop_AVrecording(filename):

    audio_thread.stop()


# Required and wanted processing of final files
def file_manager(filename):

    local_path = os.getcwd()

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")

    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")

    if os.path.exists(str(local_path) + "/temp_video2.avi"):
        os.remove(str(local_path) + "/temp_video2.avi")

    if os.path.exists(str(local_path) + "/" + filename + ".avi"):
        os.remove(str(local_path) + "/" + filename + ".avi")


class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.out = cv2.VideoWriter("./static/video.avi", fourcc, 20.0, (640, 480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

    def __del__(self):
        self.cap.release()

    def get_frame(self):

        face_cascade = cv2.CascadeClassifier(
            cv2.samples.findFile(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.samples.findFile(cv2.data.haarcascades + "haarcascade_eye.xml")
        )

        ret, frame = self.cap.read()

        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # obj = DeepFace.analyze(
            #     img_path=frame, actions=["emotion"], enforce_detection=False
            # )
            # emotion = obj["dominant_emotion"]
            # emotion_lst.append(emotion)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            # img = cv2.imread("my_plot.png")
            # # #Draw Rectangle
            # frame = cv2.rectangle(img, (100, 50), (125, 80), (0, 0, 0), 2)

            # Read logo and resize
            logo = cv2.imread("my_plot.png")
            sizex = 100
            sizey = 100
            size = 100
            logo = cv2.resize(logo, (size, size))

            # Create a mask of logo
            img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 6, 255, cv2.THRESH_BINARY)
            # Region of Interest (ROI), where we want
            # to insert logo
            roi = frame[-sizex - 10 : -10, -sizey - 10 : -10]

            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

            # Hand Recognize
            mpHands = mp.solutions.hands
            hands = mpHands.Hands()
            mpDraw = mp.solutions.drawing_utils
            imageRGB2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(imageRGB2)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:  # working with each hand
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 20:
                            cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # for (x, y, w, h) in faces:
            #     frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (230, 0, 0), 2)
            #     frame = cv2.putText(
            #         frame,
            #         emotion,
            #         (x, y),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.6,
            #         (255, 255, 255),
            #         1,
            #     )

            #     # eye Detection

            #     eyes = eye_cascade.detectMultiScale(gray)
            #     for (ex, ey, ew, eh) in eyes:
            #         # cv2.circle(
            #         #     img=frame,
            #         #     center=((ex + ew) - 20, (ey + eh) - 20),
            #         #     radius=10,
            #         #     color=(255, 0, 0),
            #         #     thickness=2,
            #         # )
            #         # frame = cv2.line(
            #         #     frame, ((ex - 20), (ey - 20)), (ex, ey - 10), (0, 255, 0), 2
            #         # )
            #         cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Record video

            if self.is_record:

                if self.out == None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self.out = cv2.VideoWriter(
                        "./static/video.avi", fourcc, 5.0, (640, 480)
                    )

                # ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)
            else:
                if self.out != None:
                    self.out.release()
                    self.out = None

            ret, jpeg = cv2.imencode(".jpg", frame)

            # self.out2.release()
            return jpeg.tobytes()

        else:
            return None

    def start_record(self):
        filename = "Default_user"
        file_manager(filename)

        start_AVrecording(filename)
        self.is_record = True
        # self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        # self.recordingThread.start()

    def stop_record(self):
        filename = "Default_user"
        stop_AVrecording(filename)
        audio_to_text()
        self.is_record = False

        # if self.recordingThread != None:
        #     self.recordingThread.stop()


import speech_recognition as sr
import pprint


def audio_to_text():
    # from pprintpp import pprint as pp
    r = sr.Recognizer()

    from os import path

    AUDIO_FILE = "recording.wav"
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.adjust_for_ambient_noise(source)
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        captured_text = r.recognize_google(audio)
        print("Google Speech Recognition transcribed your audio to 'captured_text' ")

    except sr.UnknownValueError:
        pprint.pprint("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        pprint.pprint(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )
    pp = pprint.PrettyPrinter(width=90, compact=True)

    print("Google Speech Recognition thinks you said: \n \n ")

    pp.pprint(captured_text)
    from txtai.pipeline import Transcription

    # Create transcription model
    transcribe = Transcription()

    files = ["recording.wav"]

    # Transcribe files
    transcribe = Transcription("openai/whisper-base")
    print("Executed After Model")
    for text in transcribe(files):
        print("Whisper AI transcribed your recording successfully")
        captured_text1 = text

    pp = pprint.PrettyPrinter(width=90, compact=True)

    print("Whisper AI thinks you said: \n \n ")

    pp.pprint(captured_text1)

    # Analyze Results
    import spacy
    from collections import Counter
    from string import punctuation

    # import en_core_web_sm

    # nlp = en_core_web_sm.load()
    nlp = spacy.load("en_core_web_sm")

    def get_hotwords(text):
        result = []
        pos_tag = ["PROPN", "ADJ", "NOUN"]
        doc = nlp(text.lower())
        for token in doc:
            if token.text in nlp.Defaults.stop_words or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                result.append(token.text)
        return result

    output = set(get_hotwords(captured_text))
    most_common_list = Counter(output).most_common(10)
    for item in most_common_list:
        print("Print Items Text  : ", item[0])

    # Showing Plots :
    import collections
    import numpy as np
    import pandas as pd
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from wordcloud import WordCloud, STOPWORDS

    stopwords = STOPWORDS
    wordcloud = WordCloud(
        stopwords=stopwords, background_color="white", max_words=1000
    ).generate(captured_text1)
    rcParams["figure.figsize"] = 3, 2

    plt.imshow(wordcloud)
    plt.axis("off")

    plt.savefig("my_plot.png")
    plt.show()

    filtered_words = [word for word in captured_text1.split() if word not in stopwords]
    counted_words = collections.Counter(filtered_words)
    words = []
    counts = []

    for letter, count in counted_words.most_common(10):
        words.append(letter)
        counts.append(count)

    colors = cm.rainbow(np.linspace(0, 1, 10))
    rcParams["figure.figsize"] = 20, 10
    plt.title("Top words in the headlines vs their count")
    plt.xlabel("Count")
    plt.ylabel("Words")
    plt.barh(words, counts, color=colors)
    plt.show()
