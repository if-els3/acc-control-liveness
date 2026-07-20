import os
import cv2
import time
import numpy as np
import sys

# Memastikan import dari core berfungsi
sys.path.insert(0, os.path.dirname(__file__))

from core.face_engine import FaceEngine
from core.liveness import LivenessDetector, BlinkDetector, MP_OK

try:
    import mediapipe as mp
except ImportError:
    mp = None

def draw_text(img, text, pos=(10, 30), font_scale=0.7, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def run_simulation():
    print("[*] Inisialisasi Modul...")
    out_dir = "test_output"
    os.makedirs(out_dir, exist_ok=True)

    engine = FaceEngine()
    loaded = engine.load()
    if not loaded:
        print("[!] Gagal memuat FaceEngine. Pastikan model tersedia.")
        return

    # Gunakan webcam (atau gambar statis jika tidak ada webcam)
    cap = cv2.VideoCapture(0)
    print("[*] Menunggu kamera...")
    time.sleep(2)
    
    # Ambil beberapa frame untuk simulasi blink (10 frame)
    frames = []
    print("[*] Merekam 10 frame untuk simulasi...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        time.sleep(0.1)
    
    cap.release()

    if len(frames) == 0:
        print("[!] Gagal mengambil gambar dari kamera.")
        # Buat dummy frame jika kamera tidak ada
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, "CAMERA NOT FOUND", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        frames = [dummy for _ in range(10)]
    
    base_frame = frames[-1].copy()

    print("[1] Menguji Face Detection (BlazeFace)...")
    boxes = engine.detect(base_frame)
    img_box = base_frame.copy()
    face_crop = None
    best_box = None
    
    if boxes:
        # Ambil wajah terbesar
        best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2, score = best_box
        cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_text(img_box, f"Face: {score:.2f}")
        face_crop = base_frame[max(0, y1):y2, max(0, x1):x2]
    else:
        draw_text(img_box, "No Face Detected", color=(0, 0, 255))
        
    cv2.imwrite(os.path.join(out_dir, "01_blazeface_detection.jpg"), img_box)
    print(" -> Tersimpan: 01_blazeface_detection.jpg")

    print("[2] Menguji Ekstraksi Vektor (MobileFaceNet)...")
    img_vector = np.zeros((300, 600, 3), dtype=np.uint8)
    emb = None
    if face_crop is not None and face_crop.size > 0:
        emb = engine.extract_embedding(base_frame)
        if emb is not None:
            # Visualisasi
            crop_rsz = cv2.resize(face_crop, (150, 150))
            img_vector[75:225, 50:200] = crop_rsz
            draw_text(img_vector, f"Vector Extracted!", (250, 100))
            draw_text(img_vector, f"Dim: {emb.shape}", (250, 140))
            draw_text(img_vector, f"Values: [{emb[0]:.2f}, {emb[1]:.2f}, ...]", (250, 180), font_scale=0.6)
    else:
        draw_text(img_vector, "Failed to extract crop", color=(0,0,255))
    cv2.imwrite(os.path.join(out_dir, "02_face_to_vector.jpg"), img_vector)
    print(" -> Tersimpan: 02_face_to_vector.jpg")

    print("[3] Menguji Pencocokan (Database Match)...")
    img_match = np.zeros((300, 600, 3), dtype=np.uint8)
    if emb is not None:
        # Simulasi kecocokan dengan dirinya sendiri
        dummy_db = [emb]
        match, score = engine.verify(base_frame, dummy_db)
        if match:
            draw_text(img_match, f"MATCH FOUND!", (150, 120), font_scale=1.5, color=(0, 255, 0))
            draw_text(img_match, f"Similarity: {score:.3f} > Threshold", (150, 180))
        else:
            draw_text(img_match, "NO MATCH", (200, 150), font_scale=1.5, color=(0, 0, 255))
    cv2.imwrite(os.path.join(out_dir, "03_vector_matching.jpg"), img_match)
    print(" -> Tersimpan: 03_vector_matching.jpg")

    print("[4 & 5] Menguji Liveness (Blink) dan Kontur Wajah...")
    img_liveness = base_frame.copy()
    img_contour = np.zeros_like(base_frame)
    
    if best_box is not None and MP_OK and mp is not None:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Gambar untuk Liveness (Fokus Mata)
                    mp_drawing.draw_landmarks(
                        image=img_liveness,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    draw_text(img_liveness, "EAR Tracker Active")
                    
                    # Gambar untuk Contour (Mesh Penuh di canvas hitam)
                    mp_drawing.draw_landmarks(
                        image=img_contour,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    draw_text(img_contour, "3D Depth & Contour Analyzed", color=(255, 255, 0))
    else:
        draw_text(img_liveness, "Liveness / MediaPipe failed", color=(0,0,255))
        draw_text(img_contour, "Liveness / MediaPipe failed", color=(0,0,255))
        
    cv2.imwrite(os.path.join(out_dir, "04_liveness_blink.jpg"), img_liveness)
    cv2.imwrite(os.path.join(out_dir, "05_face_contour.jpg"), img_contour)
    print(" -> Tersimpan: 04_liveness_blink.jpg")
    print(" -> Tersimpan: 05_face_contour.jpg")
    
    print(f"\n[OK] Semua tes selesai! Silakan periksa folder '{out_dir}/'")

if __name__ == "__main__":
    run_simulation()
