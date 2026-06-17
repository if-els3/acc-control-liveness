#-----FACE ENGINE-----
# pipeline: frame → blazeface → crop → mobilefacenet → embedding → cosine sim

import os
import time
import logging
import urllib.request
from typing import Optional, List, Tuple
import numpy as np
import torch

log = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

def get_blazeface_class():
    if config.BLAZEFACE_DIR not in sys.path:
        sys.path.insert(0, config.BLAZEFACE_DIR)
    try:
        from blazeface import BlazeFace
        return BlazeFace
    except ImportError as e:
        log.error(f"Gagal mengimpor BlazeFace dari {config.BLAZEFACE_DIR}: {e}")
        return None

#-----IMPORT TFLITE-----
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_OK = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_OK = True
    except ImportError:
        TFLITE_OK = False
        log.warning("TFLite tidak ada — face recognition tidak berfungsi")

#-----IMPORT OPENCV-----
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


#-----HELPER TFLITE-----

def _load_tflite(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        interp = tflite.Interpreter(
            model_path=model_path,
            num_threads=config.TFLITE_THREADS
        )
    except TypeError:
        interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _download_blazeface():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if os.path.exists(config.BLAZEFACE_MODEL):
        return True
    log.info("Mengunduh BlazeFace TFLite ...")
    try:
        urllib.request.urlretrieve(config.BLAZEFACE_URL, config.BLAZEFACE_MODEL)
        log.info("BlazeFace diunduh")
        return True
    except Exception as e:
        log.error(f"Download gagal: {e}")
        return False


#-----PREPROCESS-----

def _preprocess_blazeface(frame_bgr: np.ndarray) -> np.ndarray:
    if not CV2_OK:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rsz = cv2.resize(rgb, (config.DETECT_SIZE, config.DETECT_SIZE),
                     interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(rsz.astype(np.float32) / 255.0, 0)


def _preprocess_facenet(face_bgr: np.ndarray) -> np.ndarray:
    if not CV2_OK:
        return None
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rsz = cv2.resize(rgb, (config.FACENET_SIZE, config.FACENET_SIZE),
                     interpolation=cv2.INTER_LINEAR)
    return np.expand_dims((rsz.astype(np.float32) - 127.5) / 128.0, 0)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


#-----DECODE BLAZEFACE OUTPUT-----

def _decode_boxes(outputs, orig_w: int, orig_h: int) -> List[Tuple]:
    # decode output mediapipe face_detection_short_range.tflite
    # return list of (x1,y1,x2,y2,score)
    boxes_list = []
    try:
        scores_t = None
        boxes_t  = None
        for o in outputs:
            flat = o.flatten()
            if 0.0 <= flat.max() <= 1.0 and len(flat) < 500:
                scores_t = o
            elif o.shape[-1] == 4:
                boxes_t = o
        if scores_t is None or boxes_t is None:
            return boxes_list
        scores = scores_t.flatten()
        boxes  = boxes_t.reshape(-1, 4)
        for i, sc in enumerate(scores):
            if sc >= config.DETECT_CONFIDENCE:
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(np.clip(xmin * orig_w, 0, orig_w))
                y1 = int(np.clip(ymin * orig_h, 0, orig_h))
                x2 = int(np.clip(xmax * orig_w, 0, orig_w))
                y2 = int(np.clip(ymax * orig_h, 0, orig_h))
                if x2 > x1 and y2 > y1:
                    boxes_list.append((x1, y1, x2, y2, float(sc)))
    except Exception:
        pass
    return boxes_list


#-----FALLBACK EMBEDDING (LBPH)-----

def _lbph_embedding(face_bgr: np.ndarray) -> np.ndarray:
    # 256-dim histogram lbp, fallback jika mobilefacenet belum ada
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    rsz  = cv2.resize(gray, (64, 64))
    hist = cv2.calcHist([rsz], [0], None, [256], [0, 256]).flatten()
    return _l2_normalize(hist.astype(np.float32))


#-----MAIN ENGINE-----

class FaceEngine:

    def __init__(self):
        self.device = torch.device("cpu")
        self._bf_interp  = None
        self._fn_interp  = None
        self._bf_inp     = None
        self._bf_out     = None
        self._fn_inp     = None
        self._fn_out     = None
        self._use_lbph   = False
        self._loaded     = False

    def load(self) -> bool:
        BlazeFaceClass = get_blazeface_class()
        if BlazeFaceClass is None:
            return False
        elif not CV2_OK:
            log.error("OpenCV tidak tersedia")
            return False

        #-----PYTORCH THREAD OPTIMIZATION-----
        # set num_threads eksplisit agar pytorch tidak thrash di arm cpu
        # default pytorch bisa spawn terlalu banyak thread di rpi → latency naik
        torch.set_num_threads(int(getattr(config, "TFLITE_THREADS", 4)))
        log.info(f"PyTorch num_threads set to {torch.get_num_threads()}")

        #-----LOAD BLAZEFACE-----
        try:
            if not os.path.exists(config.BLAZEFACE_WEIGHTS):
                log.error(f"Weights tidak ditemukan: {config.BLAZEFACE_WEIGHTS}")
                return False
            
            self.net = BlazeFaceClass().to(self.device)
            self.net.load_weights(config.BLAZEFACE_WEIGHTS)
            self.net.load_anchors(config.BLAZEFACE_ANCHORS)
            self.net.eval()
            log.info("BlazeFace PyTorch dimuat.")
        except Exception as e:
            log.error(f"Gagal memuat BlazeFace: {e}")
            return False

        #-----LOAD MOBILEFACENET-----
        if TFLITE_OK:
            self._fn_interp = _load_tflite(config.MOBILEFACENET_MODEL)
            if self._fn_interp:
                self._fn_inp = self._fn_interp.get_input_details()
                self._fn_out = self._fn_interp.get_output_details()
                log.info("MobileFaceNet TFLite dimuat.")
            else:
                self._use_lbph = True
                log.warning("MobileFaceNet gagal dimuat, menggunakan LBPH.")
        else:
            self._use_lbph = True

        self._loaded = True
        return True

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple]:
        if not self._loaded or self.net is None:
            return []
            
        h_orig, w_orig = frame_bgr.shape[:2]
        frame_resized = cv2.resize(frame_bgr, (128, 128))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            detections = self.net.predict_on_image(frame_rgb)
        
        if torch.is_tensor(detections):
            detections = detections.cpu().numpy()

        results = []
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                ymin, xmin, ymax, xmax = detections[i, 0:4]
                score = detections[i, 16]
                
                if score >= config.DETECT_CONFIDENCE:
                    x1, y1 = int(xmin * w_orig), int(ymin * h_orig)
                    x2, y2 = int(xmax * w_orig), int(ymax * h_orig)
                    results.append((x1, y1, x2, y2, float(score)))
        return results

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[Tuple]:
        boxes = self.detect(frame_bgr)
        if not boxes: return None
        return max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

    def _embed_face(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr is None or face_bgr.size == 0:
            return None
        
        if self._use_lbph:
            return _lbph_embedding(face_bgr)
            
        try:
            face_norm = cv2.resize(face_bgr, (112, 112))
            face_norm = (face_norm.astype(np.float32) - 127.5) / 128.0
            face_input = np.expand_dims(face_norm, axis=0)
            
            self._fn_interp.set_tensor(self._fn_inp[0]['index'], face_input)
            self._fn_interp.invoke()
            emb = self._fn_interp.get_tensor(self._fn_out[0]['index'])[0]
            return _l2_normalize(emb.flatten().astype(np.float32))
        except Exception:
            return _lbph_embedding(face_bgr)

    def extract_embedding(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        box = self.detect_largest(frame_bgr)
        if box is None: return None
        x1, y1, x2, y2, _ = box
        face = frame_bgr[max(0,y1):y2, max(0,x1):x2]
        return self._embed_face(face)

    def verify(self, frame_bgr, stored_embeddings) -> Tuple[bool, float]:
        live_emb = self.extract_embedding(frame_bgr)
        if live_emb is None: return False, 0.0
        
        best_score = 0.0
        for stored in stored_embeddings:
            s_arr = np.array(stored, dtype=np.float32)
            score = np.dot(live_emb, s_arr)
            if score > best_score: best_score = score
            
        return best_score >= config.FACE_MATCH_THRESH, float(best_score)

    def verify_multi_frame(self, frames, stored_embeddings, min_votes=2, callback=None):
        # verifikasi dari full frames (detect + embed per frame)
        # dipakai jika liveness dinonaktifkan (tidak ada cached crops)
        votes = 0
        scores = []
        for f in frames:
            match, sc = self.verify(f, stored_embeddings)
            scores.append(sc)
            if match: votes += 1
            if callback: callback(float(sc))
        return votes >= min_votes, float(np.mean(scores)) if scores else 0.0

    def verify_crop(self, face_crop: np.ndarray,
                    stored_embeddings: list) -> Tuple[bool, float]:
        #-----VERIFY CROP-----
        # terima pre-cropped face langsung → skip blazeface re-detect
        # dipanggil oleh verify_multi_crop setelah liveness kumpulkan crops
        live_emb = self._embed_face(face_crop)
        if live_emb is None:
            return False, 0.0
        
        best_score = 0.0
        for stored in stored_embeddings:
            s_arr = np.array(stored, dtype=np.float32)
            score = np.dot(live_emb, s_arr)
            if score > best_score:
                best_score = score
        
        return best_score >= config.FACE_MATCH_THRESH, float(best_score)

    def verify_multi_crop(self, face_crops: List[np.ndarray],
                          stored_embeddings: list,
                          min_votes: int = 2,
                          callback=None) -> Tuple[bool, float]:
        #-----VERIFY MULTI CROP-----
        # terima list pre-cropped faces → langsung ke mobilefacenet
        # eliminasi redundant blazeface inference di phase 2
        # ~10x lebih cepat dari verify_multi_frame karena skip detection
        votes  = 0
        scores = []
        for crop in face_crops:
            match, sc = self.verify_crop(crop, stored_embeddings)
            scores.append(sc)
            if match:
                votes += 1
            if callback:
                callback(float(sc))
        return votes >= min_votes, float(np.mean(scores)) if scores else 0.0

    @property
    def mode(self) -> str:
        if not self._loaded:
            return "Belum dimuat"
        return "LBPH (fallback)" if self._use_lbph else "MobileFaceNet"

    def is_loaded(self): return self._loaded
