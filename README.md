# 🚨 PoseGuard: Driver Unwanted Pose Detection System

PoseGuard is a modern, deep learning-powered safety application designed to detect and log distracted or unsafe driving behaviors in real-time. Built with a robust convolutional neural network (MobileNetV2 transfer learning) and wrapped in a stunning cyberpunk-themed web interface, PoseGuard monitors drivers, flags unwanted postures, saves alert screenshots, logs incidents in an SQLite database, and plays immediate audio warning beeps.

---

## 🌟 Key Features

* **Real-time Pose Classification**: Utilizes OpenCV and TensorFlow/Keras to analyze real-time video frames (captured via webcam) and classify behaviors.
* **Video Upload Analysis**: Allows users to upload recorded driving videos and outputs a frame-by-frame highlighted assessment feed.
* **Intelligent 12-of-15 Trigger Logic**: Eliminates false positives by keeping a 15-frame history buffer; alerts are triggered only if **at least 12 of the last 15 frames** detect unsafe behavior.
* **Dynamic Auditory Alerts**: Automatically fires immediate alarm sound feedback (`beep.mp3`) when unsafe postures persist and silences it the moment safe driving is restored.
* **Database Logging & Admin Portal**: Saves a timestamped log of each distraction event alongside webcam screenshot snapshots in an SQLite database. Features an administrative portal to browse reports and captured evidence.
* **Stunning Modern UI**: A premium cyberpunk design featuring dynamic glassmorphism cards, neon glowing text headers, responsive Bootstrap layouts, and a Three.js interactive 3D particle background.

---

## 🧠 Supported Classes (State Farm Dataset)

The underlying model is trained on the Kaggle State Farm Distracted Driver Detection dataset and maps drivers to 10 distinct behaviors:

| Class Code | Class Name | Description | Status |
| :---: | :--- | :--- | :--- |
| **c0** | `Normal Pose` | Safe driving, looking straight ahead | ✅ Safe |
| **c1** | `Phone (Right Hand)` | Texting or holding a phone in the right hand | ❌ Distracted |
| **c2** | `Phone (Right hand Talking)` | Talking on a phone held to the right ear | ❌ Distracted |
| **c3** | `Phone (Left Hand)` | Texting or holding a phone in the left hand | ❌ Distracted |
| **c4** | `Phone (Left hand Talking)` | Talking on a phone held to the left ear | ❌ Distracted |
| **c5** | `Distracted....` | Operating the radio or side dashboard console | ❌ Distracted |
| **c6** | `Drinking` | Drinking water, soda, or a beverage | ❌ Distracted |
| **c7** | `Looking Back` | Reaching behind into the back seat | ❌ Distracted |
| **c8** | `Makeup` | Adjusting hair or applying makeup | ❌ Distracted |
| **c9** | `Looking Away` | Looking away or talking to a passenger | ❌ Distracted |

---

## 🛠️ Technological Stack

* **Machine Learning**: TensorFlow 2.x, Keras, MobileNetV2, NumPy
* **Computer Vision**: OpenCV (`cv2`)
* **Web Framework**: Flask, Flask-SQLAlchemy (SQLite Database)
* **Testing UI**: Streamlit
* **Security**: `bcrypt` (Secure password hashing)
* **Frontend**: HTML5, Vanilla CSS, Bootstrap 5, Javascript, Three.js, GSAP (GreenSock Animation Platform)

---

## 📂 Core Project Architecture

```
Driver-Unwanted-Pose-Detection-System_Final_Project-/
├── app.py                   # Core Flask web server, routes, and video streams
├── pose_detection.py        # Keras model inference helper and preprocessing script
├── test.py                  # Streamlit test application for offline image testing
├── requirements.txt         # Package dependencies list
├── model/
│   ├── poseguard_model.h5   # Trained MobileNetV2 neural network weights
│   ├── CNN_Model.ipynb      # Training notebook (Keras sequential)
│   └── CNN_Model_google.ipynb # Preprocessing and data generator notebook
├── static/
│   ├── css/                 # Custom styling configurations
│   ├── processed/           # Destination for processed video files
│   ├── screenshots/         # Captured screenshot snapshots of alert events
│   └── resources/
│       └── beep.mp3         # Alarm warning sound effect
└── templates/               # Flask Jinja2 HTML layout views
```

---

## 🚀 Setup & Installation

### Prerequisites
Make sure you have Python 3.10+ installed on your system.

1. **Install Dependencies**:
   Open a terminal in the project directory and install the packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Path**:
   Ensure `poseguard_model.h5` is correctly located inside the `model/` folder.

---

## 💻 Running the Applications

### 1. Running the Flask Web Portal
Start the main application web portal:
```bash
python app.py
```
* Open your browser and navigate to `http://127.0.0.1:5000`.
* Register an account and log in.
* **Admin Access**: Users whose emails end with `@poseguard.com` (e.g. `admin@poseguard.com`) are automatically granted administrative access to the Admin Portal and Reports Dashboard where screenshots and database alert histories are displayed.

### 2. Running the Streamlit Test Suite
To run quick predictions on static images (supporting JPEG, JPG, and PNG files):
```bash
streamlit run test.py
```
* Upload an image to instantly see predicted categories, bounding classifications, and confidence parameters.

---

## 🎥 Simulated Camera Angle Guidelines
Because the deep learning model was trained using inside-cabin dashboard camera angles mounted on the passenger side:
* **Camera Position**: Place your laptop/webcam **to your right** at a **45 to 60-degree angle** relative to your body.
* **Safe State**: Keep your hands down (simulating steering wheel grip) and look straight ahead (away from the camera, as if looking through the windshield). Looking directly into the camera will classify you as **Looking Away** or **Talking to Passenger**!
