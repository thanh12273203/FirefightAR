import pyrebase

# Configuration for Firebase
config = {
    "apiKey": "AIzaSyB4ncqit04UYhchiVuPclf-Cj_k-y_roQk",
    "authDomain": "fireai-f5a1a.firebaseapp.com",
    "projectId": "fireai-f5a1a",
    "storageBucket": "fireai-f5a1a.appspot.com",
    "messagingSenderId": "616759730319",
    "appId": "1:616759730319:web:107c0e8d16a360805f90fa",
    "measurementId": "G-WW47139KEX",
    "serviceAccount": "firebasesdk.json",
    "databaseURL": "https://fireai-f5a1a-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

if __name__ == '__main__':
    storage.download('videos/camera_1.mp4', 'test_video.mp4')