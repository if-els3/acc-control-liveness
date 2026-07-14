import os
import sys
import time
import logging
import numpy as np

# Atur tingkat log ke WARNING agar output benchmark bersih
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.database import Database
from core.face_engine import FaceEngine
from core.liveness import LivenessDetector
from core.camera_stream import CameraStream

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

def run_profile(duration_seconds=10, max_frames=100):
    if not PSUTIL_OK:
        print("\n" + "="*70)
        print("  [PERINGATAN] Modul 'psutil' tidak terdeteksi.")
        print("  Silakan instal terlebih dahulu untuk mengukur CPU & RAM:")
        print("  pip install psutil")
        print("="*70 + "\n")
    
    print("================================================================")
    print("  MEMULAI PROFILING KINERJA SISTEM (CPU, RAM, & FPS DETEKSI)")
    print("================================================================")
    
    print("  [1/4] Inisialisasi Database...")
    db = Database()
    
    print("  [2/4] Inisialisasi FaceEngine & Load Model...")
    face_engine = FaceEngine()
    if not face_engine.load():
        print("  [ERR] FaceEngine gagal dimuat! Cek model.")
        return
    print(f"        FaceEngine Mode: {face_engine.mode}")
    
    print("  [3/4] Inisialisasi Liveness Detector...")
    liveness_detector = LivenessDetector()
    blink_detector = liveness_detector.create_blink_detector()
    blink_detector.reset()
    
    print("  [4/4] Membuka Kamera...")
    cam = CameraStream()
    if not cam.start():
        print("  [ERR] Gagal membuka kamera!")
        return
    
    # Pre-warm psutil
    if PSUTIL_OK:
        process = psutil.Process(os.getpid())
        # Panggilan pertama cpu_percent mengembalikan 0.0, lakukan warm-up
        process.cpu_percent()
        psutil.cpu_percent()
        time.sleep(0.5)

    print("\n  >> Menjalankan benchmark... (Hadapkan wajah ke kamera jika ingin menguji deteksi)")
    print(f"     Durasi target: {duration_seconds} detik atau maks {max_frames} frame.")
    print("  -------------------------------------------------------------")

    times_frame_get = []
    times_detect = []
    times_liveness = []
    times_embed = []
    times_total = []
    
    cpu_usages_process = []
    cpu_usages_system = []
    ram_usages_process = []
    ram_usages_system = []
    
    frame_count = 0
    face_detected_count = 0
    
    t_start = time.perf_counter()
    
    try:
        while (time.perf_counter() - t_start < duration_seconds) and (frame_count < max_frames):
            t_loop_start = time.perf_counter()
            
            # 1. Capture frame
            t_cap_start = time.perf_counter()
            frame = cam.read()
            t_cap_end = time.perf_counter()
            times_frame_get.append((t_cap_end - t_cap_start) * 1000)
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            
            # 2. Face Detection
            t_det_start = time.perf_counter()
            box = face_engine.detect_largest(frame)
            t_det_end = time.perf_counter()
            times_detect.append((t_det_end - t_det_start) * 1000)
            
            # 3. Liveness Check & Embedding (jika wajah terdeteksi)
            if box is not None:
                face_detected_count += 1
                x1, y1, x2, y2, score = box
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                
                # Liveness (EAR blink detector update)
                t_live_start = time.perf_counter()
                if face_crop.size > 0:
                    blink_detector.update(face_crop)
                t_live_end = time.perf_counter()
                times_liveness.append((t_live_end - t_live_start) * 1000)
                
                # Face Embedding (MobileFaceNet / LBPH)
                t_emb_start = time.perf_counter()
                if face_crop.size > 0:
                    face_engine._embed_face(face_crop)
                t_emb_end = time.perf_counter()
                times_embed.append((t_emb_end - t_emb_start) * 1000)
            else:
                # Masukkan 0 atau abaikan agar statistik tidak terdistorsi
                times_liveness.append(0.0)
                times_embed.append(0.0)
                
            t_loop_end = time.perf_counter()
            times_total.append((t_loop_end - t_loop_start) * 1000)
            
            # 4. Measure CPU / RAM (setiap 5 frame agar tidak membebani loop utama)
            if PSUTIL_OK and (frame_count % 5 == 0):
                try:
                    # CPU process-specific (dinormalisasi dengan jumlah core CPU)
                    cpu_proc = process.cpu_percent() / psutil.cpu_count()
                    # RAM process-specific (Resident Set Size dalam MB)
                    ram_proc = process.memory_info().rss / (1024 * 1024)
                    
                    cpu_usages_process.append(cpu_proc)
                    ram_usages_process.append(ram_proc)
                    
                    # System-wide metrics
                    cpu_usages_system.append(psutil.cpu_percent())
                    ram_usages_system.append(psutil.virtual_memory().percent)
                except Exception:
                    pass
            
            # Jeda singkat untuk melepaskan giliran thread (simulasi laju frame real-time)
            time.sleep(0.02)
            
    finally:
        t_total_elapsed = time.perf_counter() - t_start
        cam.stop(force=True)
        
    print("  ✔ Benchmark Selesai!")
    print("================================================================")
    
    # Hitung Statistik Kinerja
    avg_fps = frame_count / t_total_elapsed if t_total_elapsed > 0 else 0
    avg_det_time = np.mean(times_detect) if times_detect else 0
    avg_live_time = np.mean([t for t in times_liveness if t > 0]) if [t for t in times_liveness if t > 0] else 0
    avg_emb_time = np.mean([t for t in times_embed if t > 0]) if [t for t in times_embed if t > 0] else 0
    avg_total_time = np.mean(times_total) if times_total else 0
    
    # FPS untuk deteksi wajah saja (1000 / avg_det_time)
    fps_det_only = 1000.0 / avg_det_time if avg_det_time > 0 else 0
    
    # RAM & CPU stats
    cpu_p_avg = np.mean(cpu_usages_process) if cpu_usages_process else 0
    cpu_s_avg = np.mean(cpu_usages_system) if cpu_usages_system else 0
    ram_p_avg = np.mean(ram_usages_process) if ram_usages_process else 0
    ram_s_avg = np.mean(ram_usages_system) if ram_usages_system else 0
    
    # Tampilkan Hasil di Console
    print("               HASIL PROFILING KINERJA SISTEM")
    print("================================================================")
    print(f"  Total Waktu Pengujian     : {t_total_elapsed:.2f} detik")
    print(f"  Total Frame Diproses      : {frame_count}")
    print(f"  Wajah Terdeteksi          : {face_detected_count}/{frame_count}")
    print(f"  Frame Rate Pipeline (FPS) : {avg_fps:.2f} FPS")
    print(f"  Frame Rate Deteksi Wajah  : {fps_det_only:.2f} FPS (hanya deteksi)")
    print("  -------------------------------------------------------------")
    print("  LATENCY PEMROSESAN (RATA-RATA):")
    print(f"  - Deteksi Wajah (BlazeFace) : {avg_det_time:.2f} ms")
    print(f"  - Liveness (MediaPipe EAR)  : {avg_live_time:.2f} ms")
    print(f"  - Face Recognition (Embed)  : {avg_emb_time:.2f} ms")
    print(f"  - Total Satu Siklus Loop    : {avg_total_time:.2f} ms")
    print("  -------------------------------------------------------------")
    if PSUTIL_OK:
        print("  KONSUMSI RESOURCES (CPU & RAM):")
        print(f"  - Penggunaan CPU Proses ini : {cpu_p_avg:.2f}% (dari total CPU)")
        print(f"  - Penggunaan CPU Sistem     : {cpu_s_avg:.2f}%")
        print(f"  - RAM Terpakai oleh Proses  : {ram_p_avg:.2f} MB")
        print(f"  - Penggunaan RAM Sistem     : {ram_s_avg:.2f}%")
    else:
        print("  KONSUMSI RESOURCES (CPU & RAM):")
        print("  - [!] Instal modul 'psutil' untuk menampilkan data CPU/RAM.")
    print("================================================================")
    
    # Tulis laporan ke Markdown
    report_path = "profiling_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Laporan Profiling Kinerja Sistem Kendali Akses

Laporan ini dibuat otomatis untuk menganalisis performa sistem secara riil dari sisi kecepatan pemrosesan (*frame rate*) dan konsumsi sumber daya (*CPU/RAM*).

## 1. Informasi Umum Pengujian
- **Waktu Eksekusi**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Waktu Pengujian**: {t_total_elapsed:.2f} detik
- **Total Frame Diproses**: {frame_count} frame
- **Jumlah Wajah Terdeteksi**: {face_detected_count} wajah
- **Resolusi Kamera**: {getattr(config, 'CAMERA_WIDTH', 320)}x{getattr(config, 'CAMERA_HEIGHT', 240)}

## 2. Analisis Kecepatan (*Frame Rate & Latency*)

| Tahapan Pemrosesan | Rata-rata Latensi (ms) | Maksimum Estimasi Kecepatan (FPS) |
| --- | --- | --- |
| **Deteksi Wajah (BlazeFace)** | {avg_det_time:.2f} ms | {fps_det_only:.2f} FPS |
| **Liveness Check (MediaPipe EAR)** | {avg_live_time:.2f} ms | {1000/avg_live_time if avg_live_time > 0 else 0:.2f} FPS |
| **Pengenalan Wajah (Embedding)** | {avg_emb_time:.2f} ms | {1000/avg_emb_time if avg_emb_time > 0 else 0:.2f} FPS |
| **Total Pipeline Loop** | {avg_total_time:.2f} ms | **{avg_fps:.2f} FPS** |

*Catatan: Nilai FPS Pipeline adalah kecepatan riil saat seluruh proses (pengambilan kamera + deteksi + liveness + pengenalan) berjalan bersama dalam satu siklus.*

## 3. Konsumsi Sumber Daya (*Resource Consumption*)

""")
        if PSUTIL_OK:
            f.write(f"""| Komponen Resource | Penggunaan oleh Program (Proses) | Total Penggunaan Sistem |
| --- | --- | --- |
| **CPU (Processor)** | {cpu_p_avg:.2f}% | {cpu_s_avg:.2f}% |
| **RAM (Memory)** | {ram_p_avg:.2f} MB | {ram_s_avg:.2f}% |

- **Konsumsi RAM Proses**: Program ini secara spesifik memakan memori fisik sebesar **{ram_p_avg:.2f} MB**.
- **Konsumsi CPU Proses**: Beban CPU yang dihasilkan oleh proses program ini rata-rata sebesar **{cpu_p_avg:.2f}%** dari total seluruh core CPU.
""")
        else:
            f.write("""- *Data CPU dan RAM tidak tersedia karena modul `psutil` belum terinstal saat pengujian dijalankan. Jalankan `pip install psutil` terlebih dahulu.*\n""")
            
    print(f"\n  [✔] Laporan lengkap tersimpan di: {report_path}")

if __name__ == "__main__":
    run_profile()
