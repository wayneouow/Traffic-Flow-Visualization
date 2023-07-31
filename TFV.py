import cv2
import time
import sys
import numpy as np

#path = "sample_s.mp4"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
# SCORE_THRESHOLD = 0.2
# NMS_THRESHOLD = 0.4
# CONFIDENCE_THRESHOLD = 0.4


def build_model(is_cuda):
    net = cv2.dnn.readNet("config_files/yolov5s.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture(path):
    capture = cv2.VideoCapture(path)
    return capture

def load_classes():
    class_list = []
    with open("config_files/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def pos_central(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy




import tkinter as tk
from tkinter import filedialog
import cv2
import threading
from PIL import Image, ImageTk
import urllib.request

fps_label =0

class VideoProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Traffic Flow')
        self.video_path = ""
        self.video_thread = None
        self.video_playing = False
        
        ###########################
        # BG
        ###########################
        self.blank_image = Image.open('asset/resized_default_bg.png')
        #print(type(self.blank_image))
        self.blank_image_tk = ImageTk.PhotoImage(self.blank_image)

        
        ###########################
        # VIDEO_PLAYER
        ###########################
        self.video_label = tk.Label(self.root, image=self.blank_image_tk, width=1200, height=675)
        self.video_label.pack(pady=20)

        
        ###########################
        # BTN
        ###########################
        self.file_select_button_img = ImageTk.PhotoImage(file="asset/resized_upload.png")
        self.file_select_button = tk.Button(self.root, image=self.file_select_button_img, command=self.open_file_dialog)
        self.file_select_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.process_button_img = ImageTk.PhotoImage(file="asset/resized_track.png")
        self.process_button = tk.Button(self.root, image=self.process_button_img, state=tk.DISABLED, command=self.process_video)
        self.process_button.pack(side=tk.LEFT, padx=10, pady=10)

        # self.download_button_img = ImageTk.PhotoImage(file="asset/resized_folder.png")
        # self.download_button = tk.Button(self.root, image=self.download_button_img, state=tk.DISABLED, command=self.download_frame)
        # self.download_button.pack(side=tk.BOTTOM, padx=10, pady=10)
        
        ###########################
        # SLIDER
        ###########################
        # pt size
        self.tracking_pt_size = tk.IntVar()
        self.tracking_pt_size.set(7)  

        self.slider_tracking_pt_size = tk.Scale(self.root, variable=self.tracking_pt_size, from_=1, to=15, orient=tk.HORIZONTAL, length=200, width=20)
        self.slider_tracking_pt_size.pack(side=tk.LEFT, padx=30, pady=40)

        ###########################
        # FPS
        ###########################
        self.FPS = tk.StringVar()
        self.FPS.set("")  

        self.showFPS = tk.Label(self.root, textvariable=self.FPS, font=('Arial',40,'bold'))
        self.showFPS.pack(side=tk.RIGHT, padx=50, pady=10)
        
        
        self.root.mainloop()
        

    def process_stream_url(self):
        stream_url = self.url_entry.get()
        if stream_url:
            self.process_button.config(state=tk.DISABLED)
            self.download_button.config(state=tk.DISABLED)
            self.video_thread = threading.Thread(target=self._process_stream_thread, args=(stream_url,))
            self.video_thread.start()

    def download_frame(self):
    # 下載
        urllib.request.urlretrieve("processed_video.mp4", "processed_video.mp4")

    def save_frame(self):
    # 儲存最後一個畫格
        cv2.imwrite("output.mp4", "processed_video.mp4")

    def open_file_dialog(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        self.process_button.config(state=tk.NORMAL)

    def process_video(self):
        if not self.video_playing:
            self.video_playing = True
            self.process_button.config(text="停止處理")
            self.video_thread = threading.Thread(target=self._process_video_thread)
            self.video_thread.start()
        else:
            self.video_playing = False
            self.process_button.config(text="處理影片")


    def _process_video_thread(self):
        video_capture = cv2.VideoCapture(self.video_path)

        # OpenCV影片處理
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
        net = build_model(is_cuda)

        ret, frame = video_capture.read()
        mask = np.full(frame.shape, 0).astype(np.uint8)

        start = time.time_ns()
        frame_count = 0
        total_frames = 0
        fps = -1
        
        tracking_point_size = 7
        tracking_point_size = self.tracking_pt_size.get()
        
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_filename = "processed_output.mp4"
        output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))
        # 調整影片解析度
        target_width = 1200
        target_height = 675

        # 計算解析度的比例因子
        scale_factor = min(target_width / frame_width, target_height / frame_height)

        # 計算目標解析度下的影片寬度和高度
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)

        # 建立影片撥放器 Label 的尺寸
        label_width = 1200
        label_height = 675

        # 計算影片撥放器 Label 中影片顯示的偏移量
        offset_x = (label_width - new_width) // 2
        offset_y = (label_height - new_height) // 2

        # 調整影片解析度
        while self.video_playing:
            ret, frame = video_capture.read()
            if ret:
                # OpenCV影片處理
                #frame = cv2.resize(frame,(1200,675))
                inputImage = format_yolov5(frame)
                outs = detect(inputImage, net)

                class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

                frame_count += 1
                total_frames += 1

                for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                     color = colors[int(classid) % len(colors)]
                     cv2.rectangle(frame, box, color, 2)
                     #cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                     cv2.circle(mask, pos_central(box[0], box[1], box[2], box[3]), tracking_point_size, color, -1)
                     cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

                if frame_count >= 5:
                    end = time.time_ns()
                    fps = 1000000000 * frame_count / (end - start)
                    frame_count = 0
                    start = time.time_ns()

                if fps > 0:
                    fps_label = str(round(fps,2))
                    #print(type(fps_label),fps_label)
                    #cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                #cv2.imshow("output", frame)

                processed_frame = cv2.add(frame, mask)
                output_video.write(processed_frame)
                
                # 將影片調整為目標解析度
                resized_frame = cv2.resize(processed_frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                # 在影片撥放器 Label 中顯示影片
                img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk
                self.FPS.set("FPS : "+str(round(fps,2)))

        video_capture.release()

VideoProcessor()