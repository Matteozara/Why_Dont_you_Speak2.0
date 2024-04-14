# Why don’t you speak? (WDyS)
This repo contains an imporved version of WDyS presented in: "Why don’t you speak?: a smartphone application to engage
museum visitors through Deepfake of statues" (<a href="https://dl.acm.org/doi/10.1145/3607542.3617359">paper</a>). I strongly suggest to watch that repository to better understand how the application works (<a href="https://github.com/Matteozara/Why_dont_you_speak/tree/master">git repo</a>)
<br>
<br>
This is the implementation of the paper "TITLETITLE"
<br>
The paper present two different implementations, one FAST to generate video pseudo live (WDyS_FAST) and one to generate video of high quality (WDyS_HD).
<br>
<br>
If you just want to see the final deep fakes you can go to <i>"Deep fake Results" </i> folder.
<br>
<br>
<!-- TABLE OF CONTENTS -->

### Table of contents
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#server-application">Server Application</a></li>
        <li><a href="#android-application">Android application</a></li>
      </ul>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contacts">Contacts</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
  
<br>
<br>



## Getting Started
<!--https://github.com/Matteozara/Why_dont_you_speak.git-->

First of all clone the repo:
```sh
  git clone "url repo"
  cd Why_dont_you_speak
  ```
NB: to run the project you need that both server_application and Android_application are connected to the same network.
<br>
<br>

### Server Application

To run the Server Application, first of all, set up an environment with GPU.
<br>
<br>
After, donwload the weights of the models [link here](https://drive.google.com/drive/folders/1EwbSPdOrXYlIqTS0SufuodawTS7eR-1P?usp=drive_link), put the EfficientNet weights (both), inside the <i>server_application/EfficientNet</i> folder, the Yolo8 weights inside the <i>server_application/Yolo8</i> folder, and the GAN weights inside <i>server_application/model</i> folder.
<br>
<br>
Once done, install the required packages:
```sh
  pip install torch
  pip install opencv-python
  pip install torchvision
  pip install facenet-pytorch
  pip install ffmpeg-python  or   sudo apt install ffmpeg
  pip install basicsr
  pip install gfpgan
  python -m pip install librosa  
  ```
<br>
<br>
After installed all the packages, you should go inside Server Application and run the server you want (FAST or HD):
```sh
  cd server_application
  python server_flask_HD.py or python server_flask_Fast.py
  ```
<br>
<br>
<b>PS</b>: Inside the <i>server_flask.py</i> file (both of them), you have to change the server and the port based on your network (last line of code):
<br>
app.run(host="<i>ip_server</i>", port=<i>port_number</i>, threaded=True)
<br>
<br>

### Android Application
Open the <I>WDyS</i> folder with Andoird Studio. 
<br>
<br>
Change the <i>server</i> String variable in <i>ExplanationActivity.java, MainActivity.java, AudioActivity.java</i> and <i>VideoAudioActivity.java</i>, with your server address, the one you wrote inside <i>server_flaskGPU.py</i> (you have to be sure that both Server and Andorid application run on the same network).
<br>
<br>
Run the app on your Android smartphone and try it.

<br>
<br>

## Results
The result is shown directly in the Andoird application (where there is also the possibility to save it in the gallery), but is also saved inside the <i>server_application/final_results</i> on server side.
Here there are two examples of deep fake generated, if you want to see more look inside the <i>"Deep fake Results" </i> folder.

<!--> HERE <--->

  
<br>
<br>

## License
This repository can only be used for personal/research/non-commercial purposes. However, for commercial requests, please contact us directly <a href="#contacts">contacts</a>
  
<br>
<br>

## Contacts
Matteo Zaramella - zaramella.2025806@studenti.uniroma1.it or matteozara98@gmail.com

Irene Amerini - amerini@diag.uniroma1.it

Paolo Russo - prusso@diag.uniroma1.it
  
<br>
<br>

## Acknowledgments

The code for the Deep fake generation (Generative Adversarial Network) has been taken from the [Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip), the code for the Real-ESRGAN from the [RealESRGAN repository](https://github.com/xinntao/Real-ESRGAN) and the base to improve from the [WDyS repository](https://github.com/Matteozara/Why_dont_you_speak/tree/master). We thank the authors for releasing their code and models.