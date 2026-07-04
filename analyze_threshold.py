"""
=============================================================
analyze_threshold.py — Hitung FAR/FRR/EER dari Raw Score CSV
=============================================================
Baca threshold_test_scores.csv hasil test_threshold_capture.py,
hitung FAR/FRR di banyak titik threshold (sweep matematis),
cari EER via interpolasi linear, plot grafik.

Cara pakai:
  python3 analyze_threshold.py
=============================================================
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "threshold_test_scores.csv"
OUTPUT_PLOT = "far_frr_grafik.png"

# Rentang sweep threshold (rapat, gratis karena cuma hitungan matematis)
SWEEP_MIN = 0.30
SWEEP_MAX = 0.90
SWEEP_STEP = 0.01


def load_scores(path):
    genuine, impostor = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["raw_score"])
            if row["label"] == "genuine":
                genuine.append(score)
            else:
                impostor.append(score)
    return np.array(genuine), np.array(impostor)


def compute_far_frr(genuine, impostor, thresholds):
    far_list, frr_list = [], []
    for t in thresholds:
        far = np.sum(impostor >= t) / len(impostor) * 100 if len(impostor) else 0.0
        frr = np.sum(genuine < t) / len(genuine) * 100 if len(genuine) else 0.0
        far_list.append(far)
        frr_list.append(frr)
    return np.array(far_list), np.array(frr_list)


def find_eer(thresholds, far, frr):
    """Cari titik EER via interpolasi linear di crossing FAR-FRR."""
    diff = far - frr
    sign_change = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_change) == 0:
        # gak ada crossing exact — ambil titik selisih FAR-FRR minimum
        i = np.argmin(np.abs(diff))
        return thresholds[i], (far[i] + frr[i]) / 2, False

    i = sign_change[0]
    t1, t2 = thresholds[i], thresholds[i + 1]
    d1, d2 = diff[i], diff[i + 1]
    t_eer = t1 - d1 * (t2 - t1) / (d2 - d1)
    far_eer = float(np.interp(t_eer, thresholds, far))
    return t_eer, far_eer, True


def main():
    genuine, impostor = load_scores(CSV_PATH)

    print(f"Genuine scores  (n={len(genuine)}):  {np.round(genuine, 4)}")
    print(f"Impostor scores (n={len(impostor)}): {np.round(impostor, 4)}")

    if len(genuine) == 0 or len(impostor) == 0:
        print("[!] Data genuine atau impostor kosong, cek CSV.")
        return

    thresholds = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_STEP, SWEEP_STEP)
    far, frr = compute_far_frr(genuine, impostor, thresholds)
    t_eer, eer_val, exact = find_eer(thresholds, far, frr)

    tag = "interpolasi crossing" if exact else "titik selisih minimum (no exact crossing)"
    print(f"\nEER ~= {eer_val:.2f}% pada threshold ~= {t_eer:.3f} ({tag})")
    print(f"\nRekomendasi FACE_MATCH_THRESH: {t_eer:.3f}")
    print("Cross-check ke config.py FACE_MATCH_THRESH sebelum finalisasi.")

    # Plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(thresholds, far, label="FAR (%)", color="red", linewidth=1.5)
    plt.plot(thresholds, frr, label="FRR (%)", color="blue", linewidth=1.5)
    plt.axvline(t_eer, color="gray", linestyle="--", linewidth=1,
                label=f"EER threshold ~= {t_eer:.3f}")
    plt.axhline(eer_val, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Threshold (Cosine Similarity)")
    plt.ylabel("Error Rate (%)")
    plt.title("Grafik FAR dan FRR terhadap Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nGrafik tersimpan: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
