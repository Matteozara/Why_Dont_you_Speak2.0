from flask import Flask, send_file, request
import base64
import numpy as np
from find_statue import check_single_frame_human, check_single_frame_statue, check_all_frames_human, check_all_frames_statue
from deep_fake_HD import start_generating
import torch
import cv2
#import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet  #library installed
from basicsr.utils.download_util import load_file_from_url  #library installed
from gfpgan import GFPGANer #library installed
from utils import RealESRGANer



app = Flask(__name__)

#kind of Main

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load model
if device == torch.device('cpu'):
    model = torch.jit.load("model/gan_cpu_PC_jit.pt") 
else:
    model = torch.jit.load("model/gan_gpu_jit.pt")

model.eval()

#face detector
detect = cv2.FaceDetectorYN.create("weights/YuNet/face_detection_yunet_2022mar.onnx", "", (320, 320))
img_W = int(720) #width 720  1080
img_H = int(1280) #height 1280  1920
# Set input size
detect.setInputSize((img_W, img_H))

#statue vs human classifier
model_back = torch.load('EfficientNet/EfficientNet_back.pt', map_location=device)
model_back.to(device)
model_back.eval()

#type of statue classifier
model_statue = torch.load('EfficientNet/EfficientNet_video.pt', map_location=device)
model_statue.to(device)
model_statue.eval()



#load ERSGAN and FaceHenancer
file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth']
model_path = 'weights/ESRGAN/RealESRGAN_x4plus.pth'
model_path_enhancer = 'weights/ESRGAN/GFPGANv1.3.pth'

#if not present download the weights
if not os.path.isfile(model_path):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = load_file_from_url(url=file_url[0], model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
if not os.path.isfile(model_path_enhancer):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path_enhancer = load_file_from_url(url=file_url[1], model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

#set parameters
outscale = 3.5 #4
tile = 0
tile_pad = 10
pre_pad = 0
dni_weight = None
#face_enhance = True #######################
fp32 = True
#alpha_upsampler = "realesrgan"
#ext = "auto" ########################
gpu_id = "0" #gpu-id
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4

# restorer
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=not fp32,
    gpu_id=gpu_id)

# USE ALWAYS THE FACE ENHANCER
face_enhancer = GFPGANer(
    model_path=model_path_enhancer,
    upscale=outscale,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler)




#exctract frames and create list (start + end + start)
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Extract frames from start to end
    #for i in range(frame_count):
    while 1:
        ret, frame = cap.read()
        if ret:
            #frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames.append(frame)
        else:
            cap.release()
            break  

    return frames



## STATUE ##
def start_process_statue(path, index, tipo=0):
    video = extract_frames(path)

    ## FIND STATUE ##
    if tipo == 0:
        label = check_single_frame_statue(video[0], detect, model_back, model_statue, device)
        print("run single frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_statue(video, detect, model_back, model_statue, device)
        print("run all frames done")
        if label == -1:
            return "Error"
    #set audio path (based on label found)
    audio = "Audio_wav/" + label + ".wav"
    print(audio)

    ## DEEPFAKE ##
    path_result_file = start_generating(model, detect, video, audio, face_enhancer, index)

    print("Path: ",path_result_file)
    return path_result_file



## INFO ##
def start_process_info(path, index, tipo=0):
    video = extract_frames(path)

    ## FIND STATUE ##
    if tipo == 0:
        label = check_single_frame_statue(video[0], detect, model_back, model_statue, device)
        print("run single frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_statue(video, detect, model_back, model_statue, device)
        print("run all frames done")
        if label == -1:
            return "Error"
    
    return label


## PEOPLE ##
def start_process_people(path, index, tipo=0):
    video = extract_frames(path)

    if tipo == 0:
        label = check_single_frame_human(video[0], detect)
        print("run single frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_human(video, detect)
        print("run all frames done")
        if label == -1:
            return "Error"
    if label == -1:
        return "Error"
    
    #set audio path (based on label found)
    audio = "Audio_wav/altro.wav"
    print(audio)
    path_result_file = start_generating(model, detect, video, audio, face_enhancer, index)

    print("Path: ",path_result_file)
    return path_result_file


## PEOPLE AND STATUE CUSTOM##
def start_process_custom(path, index, tipo=0):
    video = extract_frames(path)

    if tipo == 0:
        label = check_single_frame_human(video[0], detect)
        print("run sinle frame done")
        if label == -1:
            return "Error"
    else:
        label = check_all_frames_human(video, detect)
        print("run all frames done")
        if label == -1:
            return "Error"
    if label == -1:
        return "Error"

    audio = "Custom_audio/audio.wav"
    
    path_result_file = start_generating(model, detect, video, audio, face_enhancer, index)

    print("Path: ",path_result_file)
    return path_result_file


@app.route('/video_display', methods=['GET'])
def get_video():
    if os.path.isfile('final_results/final_result0.mp4'):
        return send_file('final_results/final_result0.mp4', mimetype='video/mp4')
    else:
        return "Error"


@app.route('/video_give_statue', methods=['POST'])
def json_handler_statue():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_statue(path, index, tipo) #add send file back


@app.route('/video_give_people', methods=['POST'])
def json_handler_people():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_people(path, index, tipo)


@app.route('/check_statue', methods=['POST'])
def check_statue():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_info(path, index, tipo)


#save and convert custom audio
@app.route('/send_audio', methods=['POST'])
def send_audio():
    # Get the JSON file from the request
    json_file = request.get_json()
    audio_data = json_file['audio']

    audio_data = base64.b64decode(audio_data)

    #save audio
    with open("Custom_audio/audio.3gp", "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(audio_data)

    try:
        os.system('ffmpeg -y -i Custom_audio/audio.3gp Custom_audio/audio.wav')
        return "done"
    except:
        print("Error on saving and converting the audio")
        return "Error"
    

@app.route('/process_custom', methods=['POST'])
def process_custom():
    # Get the JSON file from the request
    json_file = request.get_json()
    video_data = json_file['video']
    index = json_file['index']
    tipo = json_file['type']

    video_data = base64.b64decode(video_data)

    path = "save_video/new" + str(index) + ".mp4"
    #save video
    with open(path, "wb") as out_file:  # open for [w]riting as [b]inary
        out_file.write(video_data)

    return start_process_custom(path, index, tipo)



if __name__ == '__main__':
    app.run(host="172.20.10.14", port=3535, threaded=True) # processes=3 => up to 3 processes    server = 172.20.10.5

