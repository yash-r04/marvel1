# *Real time object detection built on Jetson Nano*
### By Yashaswini C Rao
[![Whats-App-Image-2025-11-18-at-19-59-00-e97c7fa5.jpg](https://i.postimg.cc/VkVgMJDh/Whats-App-Image-2025-11-18-at-19-59-00-e97c7fa5.jpg)](https://postimg.cc/jDyyYSV6)
---
## Edge AI!! 

Edge AI refers to executing optimized machine-learning inference directly on edge devices using localized compute resources such as NPUs, MCUs, and embedded GPUs, eliminating dependence on cloud latency and bandwidth. <br /><br />It relies on quantized, pruned, and hardware-accelerated models (e.g., TensorRT, CoreML, Edge TPU) to operate within strict power, memory, and real-time constraints. By processing sensor data in situ, Edge AI enables deterministic, low-latency control loops, enhances data privacy, and supports resilient operation in intermittently connected environments, forming the backbone of modern autonomous, industrial, and IoT systems. <br /><br />It's key advantages include reduced latency, improved data sovereignty, lower operational bandwidth costs, and high reliability at the network edge. <br /><br />A typical use-case scenario includes real-time defect detection on a factory assembly line, where an edge device performs on-device vision inference to instantly trigger responses without cloud dependency.
## sooo WHAT IS JETSON NANO??
The Jetson Nano is basically a tiny computer that looks innocent but secretly carries a mini GPU powerhouse inside it. With its 128-core NVIDIA GPU and quad-core ARM CPU, it can crunch AI models like YOLO and MobileNet in real time—while sipping just 5–10 watts of power. Think of it as a “nerdy pet robot brain”: small enough to fit in your hand, smart enough to run deep learning, and serious enough to control robots, drones, and smart cameras without breaking a sweat. Plug in a camera, load your model, and suddenly the Nano isn’t “cute” anymore—it’s detecting objects, recognizing faces, and acting like a full-blown edge AI workstation pretending to be a lunchbox-sized circuit board.
## Let's get Technical!
Find code here: 
- **Hardware Setup**
**Device:** NVIDIA Jetson Nano<br />**OS:** standard JetPack SD card image (includes Ubuntu, CUDA, cuDNN, TensorRT).<br />**Peripherals:** USB Webcam (Logitech C920) 
- **Dataset:** **MS COCO (Microsoft Common Objects in Context)**<br />The model is pre-trained on the 90-class subset of this dataset. <br />**Classes include:** People, vehicles (car, bus, bicycle), animals (dog, cat, horse), and household items (cup, bottle, chair).        
-   **Model:** **SSD-Mobilenet-v2** (Single Shot MultiBox Detector with MobileNet V2 backbone) <br />**Alternative:** SSD-Inception-v2 (Available in the downloader tool, offers higher accuracy but lower FPS).
-   **Why SSD-Mobilenet-v2?**<br />**Efficiency:** It is a lightweight architecture specifically designed for mobile and embedded devices. <br />**Speed vs. Accuracy:** It offers the best trade-off for the Jetson Nano, achieving **real-time performance (~22–25 FPS)**. Heavier models would run too slowly (low FPS) to be usable for a live camera demo.        
-   **Why TensorRT?**<br />The `jetson.inference` library runs the model through **NVIDIA TensorRT**. This optimizes the neural network layers and fuses kernels specifically for the Jetson's GPU, providing significantly faster inference than running raw TensorFlow or PyTorch on the CPU.
-   **Why MS COCO?**<br />It provides a standard, versatile set of "everyday" objects that makes the demo immediately impressive and applicable to real-world testing without requiring the user to train their own custom model first.[![1.png](https://i.postimg.cc/TPDWN0XC/1.png)](https://postimg.cc/cKdHrQCn)[![2.png](https://i.postimg.cc/k4CD1dsJ/2.png)](https://postimg.cc/jW8s2mBm)[![3.png](https://i.postimg.cc/jj0j2rGJ/3.png)](https://postimg.cc/7594tjPx)
## okayyy but where will it be used??
In an autonomous vehicle, real-time object detection on a Jetson Nano is all about speed and certainty. Because the Nano performs inference locally on its 128-core GPU, it delivers ultra-low latency—processing each camera frame in just a few tens of milliseconds. This rapid perception loop is critical: at 60 km/h, the car travels over 16 meters per second, so even a 200 ms delay (typical with cloud-based processing) means the vehicle has already moved several meters before it even realizes a pedestrian or obstacle is ahead. The Jetson Nano’s on-board computation prevents this danger by analyzing video streams instantly, enabling timely braking or steering commands. Its efficient power envelope (5–10 W) ensures this high-speed processing happens continuously without overheating or power drain. Without such local, high-throughput computation, the car would be essentially “thinking in slow motion,” and that delay could turn a small obstacle into a serious accident.
[![Whats-App-Image-2025-11-18-at-20-27-52-128bcfe2.jpg](https://i.postimg.cc/QdJ9GwWQ/Whats-App-Image-2025-11-18-at-20-27-52-128bcfe2.jpg)](https://postimg.cc/9DzFyxb0)
## Thank You
---
## BONUS: 
USER GUIDE TO JETSON NANO
-link : [nvidia official link](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup-headless)

-user kit pdf : [pdf](https://developer.nvidia.com/embedded/dlc/Jetson_Nano_Developer_Kit_User_Guide)

-----

### FOLLOW THESE STEPS FOR THE FIRST BOOT
1.Download the OS image from the website

2.Format SD card and etch the SD card using format

3.Insert sd card into jetson nano

4.Head mode:

	4.1 need monitor, keyboard, mouse

	4.2 just connect the peripherals and

	4.3 power sources: micro usb power supply (make sure jumper is removed and kept safe)

	4.4 follow first boot instructions.

5. HeadLess mode:

	5.1 does not require peripheral devices

	5.2 need to install PUTTY

	5.2.1: in putty

6. first boot:

	6.1: Review and accept NVIDIA Jetson software EULA

	6.2: Select system language, keyboard layout, and time zone

	6.3:Create username, password, and computer name

	6.4:Select APP partition size—it is recommended to use the max size suggested

7. Power Supply configuration!!

*[J48] Enables either J28 Micro-USB connector or J25 power jack as power source for the developer kit. Without a jumper, the developer kit can be powered by J28 Micro- USB connector. With a jumper, no power is drawn from J28, and the developer kit can be powered via J25 power jack.*

