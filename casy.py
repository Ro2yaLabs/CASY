import spacy
import pandas as pd
import re
import time
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import chromadb
from openai import OpenAI
import elevenlabs
import subprocess
import os
from typing import Iterator
from random import randint

from wav2lip_master import inference_yolo
from dataclasses import dataclass
from wav2lip_master import audio

from pydub import AudioSegment
from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk

import torch
from ultralytics import YOLO
from IPython.display import display

@dataclass
class Args:
    checkpoint_path = r"E:\chat\wav2lip_master\checkpoints\wav2lip.pth"
    audio = r"E:\chat\inputs\ms.wav"
    face = r"E:\chat\inputs\salma720.mp4"
    outfile = r'E:\chat\outputs\out.mp4'
    frame_path = r"E:\chat\frames"
    fps = 25
    face_det_batch_size = 16
    wav2lip_batch_size = 1
    resize_factor = 1
    crop = [0, -1, 0, -1]
    box = [-1, -1, -1, -1]
    rotate = False
    nosmooth = False
    save_frames = False
    static = False
    save_as_video = False
    img_size = 96
    pads = [0, 0, 0, 0]
    mel_step_size = 16
    device = "cpu"
    


class Chat:
    def __init__(self, file_path, i):

        self.args = Args()

        self.model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"

        self.wav2lib_model = inference_yolo.load_model(self.args.checkpoint_path)
        self.yolo_model = YOLO('wav2lip_master/yolo/best.pt')

        self.dim = 384
        self.file_path = file_path
        chroma_client = chromadb.PersistentClient(path=f"./dp/demo{i}")
        self.collection = chroma_client.get_or_create_collection(
            name="book",
            metadata={"hnsw:space": "cosine"}
        )
        full_text = self.read_docx(self.file_path)
        splitted_txt = self.splitter(full_text)
        self.model = self._encode()
        encoded_text = self.model.encode(splitted_txt, show_progress_bar=True).tolist()
        ids = [str(i) for i in range(len(encoded_text))]
        self.collection.upsert(
            documents=splitted_txt,
            embeddings=encoded_text,
            ids=ids
        )
        self.system = """
                I'll provide you with a JSON object that contains a question and the context related to it:
                {"question": the question, "context": the context}
                Please generate the answer of the provided question based on the context above.
                """
        
        api_key = "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"
        elevenlabs.set_api_key("f8b8bd17f45040b85ee67d3d0c6f0b1d")
        
        self.client = OpenAI(api_key=api_key)

        self.messages = [
            {"role": "system", "content": self.system}, 
        ]

        self.out = cv2.VideoWriter(f"temp/jhd.mp4",
                                        cv2.VideoWriter_fourcc(*'DIVX'), 25, (720, 720))

    def run(self, question):
        question_embed = self.model.encode(question)
        results = self.collection.query(
            query_embeddings=question_embed.tolist(),
            n_results=3,  
        )
        top_paragraph = ' '.join([i for i in results['documents']][0])
        prompt = '{"question": ' + question + ', "context": ' + top_paragraph + '}'

        self.messages.append(
            {"role": "user", "content": prompt}
        )

        return self.generate_audio(prompt, self.messages)

    def read_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        full_text = '\n'.join(full_text)

        return full_text

    def splitter(self, txt):
        
        chunk_size = 1000
        chunk_overlap = 200

        def length_function(text: str) -> int:
            return len(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )

        return splitter.split_text(txt)
    
    def _encode(self):
        return SentenceTransformer(self.model_id, device=self.args.device)
    
    def _get_apen_ai_answer(self, prompt, messages):
        response = self.client.chat.completions.create(
            model = "gpt-3.5-turbo-1106",
            temperature= 0,
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            txt = chunk.choices[0].delta.content
            # print(txt, end="")
            
            yield txt if txt != None else ""
            
    def stream(self, audio_stream: Iterator[bytes]) -> bytes:

        mpv_command = ["C:\\Program Files\\mpv\\mpv.exe", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = b""

        for chunk in audio_stream:
            if chunk is not None:
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore
                audio += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        return audio

    def generate_audio(self, prompt, messages):
        txt = self._get_apen_ai_answer(prompt, messages)
        audio_bytes = elevenlabs.generate(text=txt, voice="Sarah", model="eleven_monolingual_v1", stream=True)
        
        for chunk in audio_bytes:
            if chunk is not None:
                audio_segment = AudioSegment.from_file(BytesIO(chunk), format="mp3")
                audio_segment.export('temp/temp.mp3', format="mp3")
                command = 'ffmpeg -hide_banner -loglevel error -y -i {} -strict -2 {}'.format('temp/temp.mp3', 'temp/temp.wav')
                subprocess.call(command, shell=True)
                audio_path = 'temp/temp.wav'
                wav = audio.load_wav(audio_path, 16000)
                mel = audio.melspectrogram(wav)

                mel_chunks = []
                mel_idx_multiplier = 80./self.args.fps 
                i = 0
                while 1:
                    start_idx = int(i * mel_idx_multiplier)
                    if start_idx + self.args.mel_step_size > len(mel[0]):
                        mel_chunks.append(mel[:, len(mel[0]) - self.args.mel_step_size:])
                        break
                    mel_chunks.append(mel[:, start_idx : start_idx + self.args.mel_step_size])
                    i += 1

                gen = self.datagen(mel_chunks, self.args)

                root = tk.Tk()
                root.title("Image Display")
                label = tk.Label(root)
                label.pack()

                def display_image_in_tkinter(img):
                    img = Image.fromarray(img.astype('uint8'), 'RGB')
                    tkimage = ImageTk.PhotoImage(image=img)
                    label.config(image=tkimage)
                    label.image = tkimage


                for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.args.device)
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.args.device)

                    with torch.no_grad():
                        try:
                            pred = self.wav2lib_model(mel_batch, img_batch)
                        except:
                            continue
                    # pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                    
                    for p, f, c in zip(pred, frames, coords):
                        y1, y2, x1, x2 = c

                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = p
                        
                        try:
                            for i in txt:
                                print(i, end="")
                            print(end="")
                            # cv2.imshow("test", f)
                            self.out.write(f) 
                        except:
                            print(f.shape)

        self.out.release() 
        # self.stream(wav)

    def get_smoothened_boxes(self, boxes, T):
        """
        Smooth the bounding boxes over a temporal window.
        """
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images, args):
        """
        Detect faces in a batch of images using YOLO.
        """
        batch_size = args.face_det_batch_size
        # batch_size = 1
        
        while 1:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    results = self.yolo_model.predict(images[i:i + batch_size], verbose=False, device=self.args.device)
                    try:
                        if self.args.device == "cuda":
                            boxes = results[0].boxes.cpu().xyxy[0].tolist()
                        else:
                            boxes = results[0].boxes.xyxy[0].tolist()
                        predictions.append(boxes)
                    except Exception as e:
                        cv2.imwrite(f"temp/faulty_frame{randint(0, 10000)}.jpg", images[0])
                        print("face not detected")
                    
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = args.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            
            y1 = max(0, int(rect[1]) - pady1)
            y2 = min(image.shape[0], int(rect[3]) + pady2)
            x1 = max(0, int(rect[0]) - padx1)
            x2 = min(image.shape[1], int(rect[2]) + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not args.nosmooth: 
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results 

    def datagen(self, mels, args):
        """
        Data generator for processing batches.
        """
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        reader = self.read_frames()
        t = []
        prev = None
        for i, m in enumerate(mels):
            try:
                frame_to_save = next(reader)
            except StopIteration:
                reader = self.read_frames()
                frame_to_save = next(reader)
            
            try:
                prev = self.face_detect([frame_to_save], args)[0]
                face, coords = prev
            except:
                face, coords = prev

            face = cv2.resize(face, (args.img_size, args.img_size))
                
            if i%10000 == 0:
                cv2.imwrite(f"test{i}.jpg", face)

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, args.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        # ss = sum(t)
        # print(f"avg: {ss/len(t)}")
        # print(f"total for face detection: {ss}")

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def read_frames(self):
        """
        Read frames from a folder of image files.
        """
        
        image_files = [f for f in os.listdir(self.args.frame_path) if f.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
        image_files.sort()

        for image_file in image_files:
            image_path = os.path.join(self.args.frame_path, image_file)
            frame = cv2.imread(image_path)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame

