"""
=============================================================
perf/runner.py  --  Live Performance Collector (--perf mode)
=============================================================
Diaktifkan oleh flag `--perf` di main.py.

Cara kerja:
  • PerfCollector.record_attempt() dipanggil di akhir setiap
    percobaan autentikasi (di _proses_akses di menus/access.py).
  • Selama sistem berjalan, setiap percobaan direkam ke memori.
  • Saat sistem ditutup, generate_report() dipanggil otomatis
    dari finally-block di main.py untuk membuat:
    
    perf_output/<timestamp>/
      csv/
        auth_log.csv              <- log utama tiap percobaan
        liveness_apcer.csv        <- template APCER/BPCER/ACER
        face_far_frr.csv          <- template FAR/FRR/EER
        pipeline_summary.csv      <- ringkasan performa pipeline
      profiling_report.md         <- laporan lengkap

Kolom auth_log.csv (untuk Tugas Akhir):
  session_id, timestamp, uid, nama, status,
  t_rfid_s, t_face_detect_s, t_liveness_s, t_verify_s, t_total_s,
  liveness_required, liveness_detected, liveness_result, liveness_score,
  face_similarity, face_match, face_threshold,
  ear_min, ear_max, ear_avg,
  label (untuk APCER: live/spoof — diisi manual)
=============================================================
"""

import os
import csv
import time
import logging
import datetime
import threading

log = logging.getLogger(__name__)

# ── Singleton instance ────────────────────────────────────────────────────────
_collector: "PerfCollector | None" = None


def get_collector() -> "PerfCollector | None":
    """Return active PerfCollector atau None jika --perf tidak aktif."""
    return _collector


def init_collector() -> "PerfCollector":
    """Inisialisasi dan simpan singleton PerfCollector."""
    global _collector
    _collector = PerfCollector()
    return _collector


# ── PerfCollector ─────────────────────────────────────────────────────────────

class PerfCollector:
    """
    Merekam setiap percobaan autentikasi secara pasif selama sistem berjalan.
    Thread-safe (RLock digunakan untuk proteksi list).
    """

    def __init__(self):
        self._lock      = threading.RLock()
        self._attempts: list[dict] = []
        self._session_counter = 0
        self._start_time = datetime.datetime.now()
        log.info("[PERF] PerfCollector aktif — merekam tiap percobaan autentikasi.")
        print("\n  [PERF] Mode Pengujian Aktif — data tiap percobaan akan direkam.")

    # ------------------------------------------------------------------
    def record_attempt(self,
                       uid: str,
                       nama: str,
                       status: str,
                       # timing (seconds)
                       t_rfid: float        = 0.0,
                       t_face_detect: float = 0.0,
                       t_liveness: float    = 0.0,
                       t_verify: float      = 0.0,
                       # liveness
                       liveness_required: int    = 0,
                       liveness_detected: int    = 0,
                       liveness_result: str      = "",
                       liveness_score: float     = 0.0,
                       # face recognition
                       face_similarity: float    = 0.0,
                       face_match: bool          = False,
                       face_threshold: float     = 0.0,
                       # EAR stats (opsional, dari blink_detector)
                       ear_min: float = 0.0,
                       ear_max: float = 0.0,
                       ear_avg: float = 0.0,
                       ):
        """
        Simpan satu record percobaan autentikasi.
        Dipanggil dari _proses_akses() di menus/access.py.
        """
        with self._lock:
            self._session_counter += 1
            record = {
                "session_id"         : self._session_counter,
                "timestamp"          : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "uid"                : uid,
                "nama"               : nama,
                "status"             : status,
                # timing
                "t_rfid_s"           : round(t_rfid, 3),
                "t_face_detect_s"    : round(t_face_detect, 3),
                "t_liveness_s"       : round(t_liveness, 3),
                "t_verify_s"         : round(t_verify, 3),
                "t_total_s"          : round(t_rfid + t_face_detect + t_liveness + t_verify, 3),
                # liveness
                "liveness_required"  : liveness_required,
                "liveness_detected"  : liveness_detected,
                "liveness_result"    : liveness_result,
                "liveness_score"     : round(liveness_score, 4),
                # face recognition
                "face_similarity"    : round(face_similarity, 4),
                "face_match"         : int(face_match),
                "face_threshold"     : round(face_threshold, 4),
                # EAR
                "ear_min"            : round(ear_min, 4),
                "ear_max"            : round(ear_max, 4),
                "ear_avg"            : round(ear_avg, 4),
                # kolom untuk analisis manual
                "label"              : "",   # isi: live / spoof (untuk APCER)
                "notes"              : "",
            }
            self._attempts.append(record)
            # print inline summary tiap percobaan
            sim_str  = f"sim={face_similarity*100:.1f}%" if face_similarity else ""
            live_str = f"lv={liveness_result}({liveness_detected}/{liveness_required}blink)" if liveness_result else ""
            print(f"  [PERF #{self._session_counter}] {uid} | {nama} | {status} | "
                  f"{sim_str} | {live_str} | total={t_rfid+t_face_detect+t_liveness+t_verify:.2f}s")

    # ------------------------------------------------------------------
    def generate_report(self):
        """
        Dipanggil dari finally-block main.py saat sistem ditutup.
        Membuat folder output, CSV, dan profiling_report.md.
        """
        with self._lock:
            attempts = list(self._attempts)

        ts        = datetime.datetime.now()
        ts_str    = ts.strftime("%Y-%m-%d %H:%M:%S")
        ts_folder = ts.strftime("%Y%m%d_%H%M%S")
        n         = len(attempts)

        base_dir = "perf_output"
        out_dir  = os.path.join(base_dir, ts_folder)
        csv_dir  = os.path.join(out_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  [PERF] Membuat laporan pengujian ...")
        print(f"  Output : {out_dir}/")
        print(f"  Total  : {n} percobaan direkam")
        print(f"{'='*60}")

        if n == 0:
            print("  [PERF] Tidak ada percobaan yang direkam — laporan kosong.")
            _write_empty_report(out_dir, ts_str)
            return

        # ── 1. auth_log.csv ─────────────────────────────────────────
        _write_auth_log(csv_dir, attempts)

        # ── 2. liveness_apcer.csv ────────────────────────────────────
        _write_liveness_apcer(csv_dir, attempts)

        # ── 3. face_far_frr.csv ──────────────────────────────────────
        _write_face_far_frr(csv_dir, attempts)

        # ── 4. pipeline_summary.csv ──────────────────────────────────
        _write_pipeline_summary(csv_dir, attempts)

        # ── 5. profiling_report.md ───────────────────────────────────
        _write_md_report(out_dir, ts_str, attempts, self._start_time)

        print(f"\n  [PERF] Selesai! Semua output ada di: ./{out_dir}/")
        print(f"{'='*60}\n")


# ── CSV Writers ───────────────────────────────────────────────────────────────

_AUTH_LOG_HEADER = [
    "session_id", "timestamp", "uid", "nama", "status",
    "t_rfid_s", "t_face_detect_s", "t_liveness_s", "t_verify_s", "t_total_s",
    "liveness_required", "liveness_detected", "liveness_result", "liveness_score",
    "face_similarity", "face_match", "face_threshold",
    "ear_min", "ear_max", "ear_avg",
    "label", "notes",
]


def _write_auth_log(csv_dir: str, attempts: list[dict]):
    """
    auth_log.csv — log utama seluruh percobaan.
    Ini adalah CSV inti yang dipakai untuk analisis akurasi,
    performa, dan keamanan (liveness) di dokumen TA.
    """
    path = os.path.join(csv_dir, "auth_log.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_AUTH_LOG_HEADER)
        w.writeheader()
        for a in attempts:
            w.writerow({k: a.get(k, "") for k in _AUTH_LOG_HEADER})
    print(f"  [CSV] {path}")


def _write_liveness_apcer(csv_dir: str, attempts: list[dict]):
    """
    liveness_apcer.csv — data untuk menghitung APCER/BPCER/ACER.
    Kolom 'label' (live/spoof) perlu diisi manual setelah pengujian.
    """
    path = os.path.join(csv_dir, "liveness_apcer.csv")
    header = [
        "session_id", "timestamp", "nama", "status",
        "liveness_required", "liveness_detected", "liveness_result",
        "liveness_score", "ear_min", "ear_max", "ear_avg",
        "label",          # WAJIB DIISI: live / spoof
        "apcer_fp",       # 1 jika spoof tapi diterima, 0 lainnya
        "bpcer_fn",       # 1 jika live tapi ditolak, 0 lainnya
        "notes",
    ]
    lv_attempts = [a for a in attempts if a.get("liveness_required", 0) > 0]
    rows = []
    for a in lv_attempts:
        # placeholder: apcer/bpcer diisi setelah label diisi manual
        rows.append([
            a["session_id"], a["timestamp"], a["nama"], a["status"],
            a["liveness_required"], a["liveness_detected"],
            a["liveness_result"], a["liveness_score"],
            a["ear_min"], a["ear_max"], a["ear_avg"],
            a.get("label", ""),  # kosong — isi manual: live/spoof
            "",  # apcer_fp — isi setelah label
            "",  # bpcer_fn — isi setelah label
            a.get("notes", ""),
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # Tambahkan baris rumus di bawah (sebagai komentar teks)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(["# RUMUS (hitung setelah isi kolom label):"])
        w.writerow(["# APCER = SUM(apcer_fp) / COUNT(label=spoof)"])
        w.writerow(["# BPCER = SUM(bpcer_fn) / COUNT(label=live)"])
        w.writerow(["# ACER  = (APCER + BPCER) / 2"])

    print(f"  [CSV] {path}  ({len(lv_attempts)} percobaan liveness)")


def _write_face_far_frr(csv_dir: str, attempts: list[dict]):
    """
    face_far_frr.csv — data untuk menghitung FAR/FRR/EER/Accuracy.
    Kolom 'label' (genuine/impostor) perlu diisi manual.
    """
    path = os.path.join(csv_dir, "face_far_frr.csv")
    header = [
        "session_id", "timestamp", "nama", "status",
        "face_similarity", "face_threshold", "face_match",
        "label",       # WAJIB DIISI: genuine / impostor
        "tp", "tn", "fp", "fn",   # isi setelah label
        "notes",
    ]
    face_attempts = [a for a in attempts
                     if a.get("face_similarity", 0) > 0 or a.get("status") in
                     ("GRANTED", "DENIED_FACE")]
    rows = []
    for a in face_attempts:
        rows.append([
            a["session_id"], a["timestamp"], a["nama"], a["status"],
            a["face_similarity"], a["face_threshold"], a["face_match"],
            a.get("label", ""),  # kosong — isi manual: genuine/impostor
            "", "", "", "",       # tp/tn/fp/fn
            a.get("notes", ""),
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(["# RUMUS (hitung setelah isi kolom label):"])
        w.writerow(["# FAR      = SUM(fp) / COUNT(label=impostor)"])
        w.writerow(["# FRR      = SUM(fn) / COUNT(label=genuine)"])
        w.writerow(["# EER      = nilai threshold saat FAR == FRR"])
        w.writerow(["# Accuracy = (TP + TN) / total"])

    print(f"  [CSV] {path}  ({len(face_attempts)} percobaan face-verify)")


def _write_pipeline_summary(csv_dir: str, attempts: list[dict]):
    """
    pipeline_summary.csv — statistik performa tiap tahapan pipeline.
    """
    import statistics

    def _stats(vals):
        vals = [v for v in vals if v > 0]
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        return (round(statistics.mean(vals), 3),
                round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 3),
                round(min(vals), 3),
                round(max(vals), 3))

    keys = ["t_rfid_s", "t_face_detect_s", "t_liveness_s", "t_verify_s", "t_total_s"]
    labels = ["RFID Scan", "Face Detect (BlazeFace)", "Liveness (EAR)",
              "Face Verify (MobileFaceNet)", "Total Pipeline"]

    path = os.path.join(csv_dir, "pipeline_summary.csv")
    header = ["tahapan", "n_samples", "avg_s", "std_s", "min_s", "max_s",
              "avg_ms", "std_ms", "min_ms", "max_ms"]
    rows = []
    for key, label in zip(keys, labels):
        vals = [a[key] for a in attempts if key in a]
        avg, std, mn, mx = _stats(vals)
        n = len([v for v in vals if v > 0])
        rows.append([
            label, n,
            avg, std, mn, mx,
            round(avg*1000, 1), round(std*1000, 1),
            round(mn*1000, 1),  round(mx*1000, 1),
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  [CSV] {path}")


# ── Markdown Report ───────────────────────────────────────────────────────────

def _write_md_report(out_dir: str, ts_str: str, attempts: list[dict],
                     start_time: datetime.datetime):
    import statistics
    import config

    n = len(attempts)
    n_granted     = sum(1 for a in attempts if a["status"] == "GRANTED")
    n_denied_face = sum(1 for a in attempts if a["status"] == "DENIED_FACE")
    n_denied_spoof= sum(1 for a in attempts if a["status"] == "DENIED_SPOOF")
    n_denied_rfid = sum(1 for a in attempts if a["status"] == "DENIED_RFID")
    n_error       = sum(1 for a in attempts if a["status"] == "ERROR")

    def _avg(key):
        vals = [a[key] for a in attempts if a.get(key, 0) > 0]
        return round(statistics.mean(vals), 3) if vals else 0.0

    def _f(v, d=2):
        return f"{v:.{d}f}" if v else "—"

    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = int(elapsed % 60)

    lines = [
        "# Laporan Pengujian Sistem Kendali Akses",
        "",
        f"> Mode: `--perf` (Pasif — data dikumpulkan selama sistem berjalan)  ",
        f"> Sistem mulai : {start_time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Laporan dibuat: {ts_str}  ",
        f"> Durasi pengujian: {h:02d}:{m:02d}:{s:02d}",
        "",
        "---",
        "",
        "## 1. Konfigurasi Sistem",
        "| Parameter | Nilai |",
        "| --- | --- |",
        f"| Face Match Threshold | {config.FACE_MATCH_THRESH} |",
        f"| EAR Threshold (tutup) | {config.BLINK_EAR_THRESHOLD} |",
        f"| Liveness Enabled | {config.LIVENESS_ENABLED} |",
        f"| Liveness Duration | {config.LIVENESS_DURATION} s |",
        f"| Min/Max Blink | {config.LIVENESS_BLINK_MIN_COUNT} / {config.LIVENESS_BLINK_MAX_COUNT} |",
        f"| Enroll Frames | {config.ENROLL_FRAMES} |",
        "",
        "---",
        "",
        "## 2. Ringkasan Percobaan",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| Total Percobaan | {n} |",
        f"| GRANTED | {n_granted} ({n_granted/n*100:.1f}%) |" if n else "| GRANTED | 0 |",
        f"| DENIED_FACE | {n_denied_face} |",
        f"| DENIED_SPOOF (Liveness) | {n_denied_spoof} |",
        f"| DENIED_RFID | {n_denied_rfid} |",
        f"| ERROR | {n_error} |",
        "",
        "---",
        "",
        "## 3. Performa Pipeline (Rata-rata per Tahapan)",
        "| Tahapan | Avg (ms) | Keterangan |",
        "| --- | --- | --- |",
        f"| RFID Scan | {_avg('t_rfid_s')*1000:.1f} | Waktu menunggu + baca kartu |",
        f"| Face Detect (BlazeFace) | {_avg('t_face_detect_s')*1000:.1f} | Deteksi wajah pertama kali |",
        f"| Liveness EAR | {_avg('t_liveness_s')*1000:.1f} | Durasi sesi liveness |",
        f"| Face Verify (MobileFaceNet) | {_avg('t_verify_s')*1000:.1f} | Embedding + cosine similarity |",
        f"| **Total Pipeline** | **{_avg('t_total_s')*1000:.1f}** | Dari tap RFID sampai hasil |",
        "",
        "---",
        "",
        "## 4. Akurasi Pengenalan Wajah",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| Avg Cosine Similarity | {_avg('face_similarity'):.4f} |",
        f"| Face Threshold | {config.FACE_MATCH_THRESH} |",
        "",
        "> **FAR / FRR / EER / Accuracy**: Isi kolom `label` (genuine/impostor)",
        "> di `csv/face_far_frr.csv` lalu hitung dengan rumus di bawah file.",
        "",
        "---",
        "",
        "## 5. Keamanan Liveness (Anti-Spoofing)",
        "| Metrik | Nilai |",
        "| --- | --- |",
        f"| Avg Liveness Score | {_avg('liveness_score'):.4f} |",
        f"| DENIED_SPOOF | {n_denied_spoof} |",
        "",
        "> **APCER / BPCER / ACER**: Isi kolom `label` (live/spoof)",
        "> di `csv/liveness_apcer.csv` lalu hitung dengan rumus di bawah file.",
        "",
        "---",
        "",
        "## 6. File Output",
        "| File | Keterangan |",
        "| --- | --- |",
        "| `csv/auth_log.csv` | Log lengkap tiap percobaan (gunakan ini untuk analisis utama) |",
        "| `csv/liveness_apcer.csv` | Template APCER/BPCER/ACER — isi kolom `label` manual |",
        "| `csv/face_far_frr.csv` | Template FAR/FRR/EER — isi kolom `label` manual |",
        "| `csv/pipeline_summary.csv` | Statistik performa tiap tahapan pipeline |",
        "",
        "---",
        "",
        "## 7. Panduan Pengisian Data Manual",
        "",
        "### Untuk Analisis Liveness (APCER/BPCER/ACER)",
        "1. Buka `csv/liveness_apcer.csv`",
        "2. Isi kolom `label` dengan nilai `live` atau `spoof` untuk tiap baris",
        "3. Isi `apcer_fp = 1` jika label=spoof tapi `liveness_result=LIVE` (false accept)",
        "4. Isi `bpcer_fn = 1` jika label=live tapi `liveness_result=SPOOF` (false reject)",
        "5. Hitung:",
        "```",
        "APCER = SUM(apcer_fp) / COUNT(label=spoof)",
        "BPCER = SUM(bpcer_fn) / COUNT(label=live)",
        "ACER  = (APCER + BPCER) / 2",
        "```",
        "",
        "### Untuk Analisis Wajah (FAR/FRR/EER/Accuracy)",
        "1. Buka `csv/face_far_frr.csv`",
        "2. Isi kolom `label` dengan nilai `genuine` atau `impostor` untuk tiap baris",
        "3. Isi `tp/tn/fp/fn` berdasarkan `face_match` vs `label`",
        "4. Hitung:",
        "```",
        "FAR      = FP / COUNT(label=impostor)  # impostor lolos = False Accept",
        "FRR      = FN / COUNT(label=genuine)   # genuine ditolak = False Reject",
        "EER      = threshold saat FAR == FRR",
        "Accuracy = (TP + TN) / total",
        "```",
        "",
    ]

    md_path = os.path.join(out_dir, "profiling_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [MD]  {md_path}")


def _write_empty_report(out_dir: str, ts_str: str):
    md_path = os.path.join(out_dir, "profiling_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Laporan Pengujian\n\n> Dibuat: {ts_str}\n\n"
                "Tidak ada percobaan yang direkam.\n")
    print(f"  [MD]  {md_path}")
