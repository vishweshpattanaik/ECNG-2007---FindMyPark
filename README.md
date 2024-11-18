# 🚗 FindMyPark - A Parking Detection System  

FindMyPark is a parking detection system that uses a live video stream to identify available parking spots in a university parking lot. The system leverages YOLO V8 for real-time detection, color-coding parking spaces as **red** (unavailable) and **green** (available). The detected spaces are displayed on a user-friendly mobile app built with FlutterFlow (not inlcuded).  

---

## 🌟 Features  

- **Real-Time Detection:** Uses live-stream video input to monitor parking spaces.  
- **Color-Coded Indicators:** Red 🟥 for unavailable, Green 🟩 for available spots.  
- **Mobile Integration:** Parking data is displayed through a responsive Flutter-based app.  
- **Environmentally Friendly:** Helps reduce emissions by minimizing search time.  

---

## 🚀 How It Works  

1. **Live Feed Input:** The system processes live video streams from cameras.  
2. **Detection Algorithm:** YOLO V8 detects cars in the parking space and their occupancy status.  
3. **Data Transmission:** Detected availability is sent to a cloud-based database using Firebase.  
4. **User App:** Users view parking availability through the FindMyPark mobile app.  

---

## 🔧 Technologies Used
1. **YOLO V8:** For object detection.
2. **OpenCV:** To process video streams.
3. **Flutter:** For the mobile application.
4. **Firebase:** For cloud-based real-time data storage.

## 🧑‍💻 Authors and Contributors 
This is a UWI ECNG 2007 -  Computer Systems and Software Design (Semester I- 2024/2025) project done by:

Vishwesh Pattanaik, Renita Bharatsingh, Kiara Creed, Daniel George and Johnathan Manzano
## Mentored by:
Kashanna Joseph and Jovan Moolah
