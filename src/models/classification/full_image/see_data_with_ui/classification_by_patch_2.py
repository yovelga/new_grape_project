import os, sys, csv, logging, time
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import joblib
import cv2
import spectral as spy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSlider, QHBoxLayout, QStatusBar,
    QSpinBox, QDoubleSpinBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[5]

# ===== LDAModel shim (for pickles that reference __main__.LDAModel) =====
class LDAModel:
    """Shim for legacy pickles. Delegates predict_proba to an inner estimator."""
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        self.__dict__.update(state)
    def _inner(self):
        for attr in ("model", "clf", "estimator_", "lda", "classifier", "classifier_"):
            est = getattr(self, attr, None)
            if est is not None:
                return est
        return None
    def predict_proba(self, X):
        est = self._inner()
        if est is None or not hasattr(est, "predict_proba"):
            raise AttributeError("LDAModel shim: no inner estimator with predict_proba")
        return est.predict_proba(X)

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HSI.Patch")
if os.environ.get("HSI_DEBUG") == "1":
    logger.setLevel(logging.DEBUG)

# ===== Config =====
DEFAULT_LDA_PATH = r"C:\\Users\\yovel\\Desktop\\Grape_Project\\classification\\pixel_level\\simple_classification_leave_one_out\\comare_all_models\\models\\LDA_Balanced.pkl"
DEFAULT_SEARCH_FOLDER = str(_PROJECT_ROOT / "dest")

# percent buckets -> BGR colors (yellow to red gradient)
GRID_COLOR_BUCKETS = [
    (40, (0, 0, 255)),     # >=40% : bright red
    (30, (0, 69, 255)),    # >=30% : red-orange
    (20, (0, 140, 255)),   # >=20% : orange
    (10, (0, 215, 255)),   # >=10% : yellow-orange
    (5,  (0, 255, 255)),   # >=5%  : yellow
]

# ===== Small helpers =====

def _predict_proba_any(model, X):
    """Try common attributes to find a classifier with predict_proba."""
    for attr in (None, "estimator_", "clf", "model", "lda", "tree", "classifier", "classifier_"):
        m = getattr(model, attr, model) if attr else model
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)
    raise AttributeError("No predict_proba on provided model")


def _classes_any(model):
    for attr in (None, "estimator_", "clf", "model", "lda", "tree", "classifier", "classifier_"):
        m = getattr(model, attr, model) if attr else model
        if hasattr(m, "classes_"):
            return getattr(m, "classes_")
    return np.array([0, 1])


def find_scaler(path_dir: str, exclude: Optional[List[str]] = None) -> Optional[str]:
    exclude = set(exclude or [])
    cands = [f for f in os.listdir(path_dir) if f.lower().endswith((".joblib", ".pkl")) and f not in exclude]
    for name in cands:  # prefer names that look like a scaler
        if any(k in name.lower() for k in ("scaler", "standard", "std")):
            p = os.path.join(path_dir, name)
            try:
                if hasattr(joblib.load(p), "transform"): return p
            except Exception:
                pass
    # fallback: any obj with transform
    for name in cands:
        p = os.path.join(path_dir, name)
        try:
            if hasattr(joblib.load(p), "transform"): return p
        except Exception:
            pass
    return None


def load_model_and_scaler(lda_path: str, scaler_path: Optional[str] = None) -> Tuple[object, Optional[object], int]:
    logger.info("Loading model: %s", lda_path)
    lda = joblib.load(lda_path)
    if scaler_path is None:
        scaler_path = find_scaler(os.path.dirname(lda_path), [os.path.basename(lda_path)])
    scaler = joblib.load(scaler_path) if scaler_path else None
    classes = _classes_any(lda)
    pos_idx = int(np.where(classes == 1)[0][0]) if 1 in set(classes) and len(classes) > 1 else (1 if len(classes) > 1 else 0)
    logger.info("Model loaded. classes=%s | pos_idx=%d | scaler=%s", np.array2string(classes), pos_idx, bool(scaler))
    return lda, scaler, pos_idx


def load_cube(hdr_path: str) -> np.ndarray:
    t0 = time.perf_counter()
    dat_path = hdr_path.replace(".hdr", ".dat")
    cube = np.array(spy.envi.open(hdr_path, dat_path).load())
    logger.info("Loaded cube: %s (%.2fs) shape=%s", hdr_path, time.perf_counter() - t0, tuple(cube.shape))
    return cube


def per_pixel_probs(cube: np.ndarray, lda: object, scaler: Optional[object], pos_idx: int) -> np.ndarray:
    t0 = time.perf_counter()
    H, W, C = cube.shape
    X = np.nan_to_num(cube.reshape(-1, C), copy=False)
    scaled = False
    if (scaler is not None and hasattr(scaler, "transform")):
        X = scaler.transform(X)
        scaled = True
    probs = _predict_proba_any(lda, X)[:, pos_idx].reshape(H, W)
    logger.info("Per-pixel probs: scaled=%s | time=%.2fs | min=%.4f max=%.4f mean=%.4f", scaled, time.perf_counter()-t0, float(probs.min()), float(probs.max()), float(probs.mean()))
    return probs


def iter_grid(H: int, W: int, cell: int):
    for r0 in range(0, H, cell):
        r1 = min(r0 + cell, H)
        for c0 in range(0, W, cell):
            c1 = min(c0 + cell, W)
            yield r0, c0, r1, c1


def analyze_grid(prob_map: np.ndarray, cell: int, pix_thr: float) -> List[Dict]:
    H, W = prob_map.shape
    logger.info("Analyzing grid: H=%d W=%d cell=%d thr=%.2f", H, W, cell, pix_thr)
    out = []
    for r0, c0, r1, c1 in iter_grid(H, W, cell):
        patch = prob_map[r0:r1, c0:c1]
        n = patch.size
        k = int((patch >= pix_thr).sum())
        pct = 100.0 * k / max(1, n)
        out.append({
            "row0": r0, "col0": c0, "row1": r1, "col1": c1,
            "cell": cell, "area": n, "count_cracked": k,
            "percent_cracked": pct, "pix_thr": pix_thr
        })
        logger.debug("Grid r0=%d c0=%d r1=%d c1=%d | n=%d cracked=%d (%.1f%%)", r0, c0, r1, c1, n, k, pct)
    percents = [c["percent_cracked"] for c in out]
    mean_p = float(np.mean(percents)) if percents else 0.0
    p50 = float(np.median(percents)) if percents else 0.0
    p90 = float(np.percentile(percents, 90)) if percents else 0.0
    logger.info("Grid summary: patches=%d | mean=%.2f%% p50=%.2f%% p90=%.2f%%", len(out), mean_p, p50, p90)
    return out


def color_for_percent(p: float):
    for thr, col in GRID_COLOR_BUCKETS:
        if p >= thr:
            return col
    return None


def overlay_on_band(band_img: np.ndarray, grid_stats: List[Dict], alpha: float = 0.35) -> np.ndarray:
    rgb = cv2.cvtColor(band_img, cv2.COLOR_GRAY2RGB)
    over = rgb.copy()
    for c in grid_stats:
        col = color_for_percent(c["percent_cracked"])
        if col:
            r0, c0, r1, c1 = c["row0"], c["col0"], c["row1"], c["col1"]
            cv2.rectangle(over, (c0, r0), (c1 - 1, r1 - 1), col, -1)
            cv2.rectangle(over, (c0, r0), (c1 - 1, r1 - 1), (0, 0, 0), 1)
    merged = cv2.addWeighted(over, alpha, rgb, 1 - alpha, 0)
    return merged

# ===== Legend rendering (outside the image) =====

def render_legend_image(buckets: List[Tuple[int, Tuple[int, int, int]]]) -> np.ndarray:
    # Create a small horizontal legend image (RGB)
    w, h, pad = 240, 140, 12
    img = np.full((h, w, 3), 240, np.uint8)
    x, y = 12, 20
    box_w, box_h = 28, 18
    for thr, bgr in buckets:
        rgb = (bgr[2], bgr[1], bgr[0])
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), rgb, -1)
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), 1)
        label = f">= {thr}%"
        cv2.putText(img, label, (x + box_w + 8, y + box_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + box_w + 8, y + box_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += box_h + pad
    title = "Cracked ratio by grid"
    cv2.putText(img, title, (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, title, (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 1, cv2.LINE_AA)
    return img

# ===== CSV export =====

def export_grid_csv(grid_stats: List[Dict], context: Dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = f"grid_percent_{context['folder_main']}_{context['folder_date']}_b{context['band']}_c{context['cell']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    fp = os.path.join(out_dir, fn)
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id","folder_main","folder_date","band","cell","pix_thr","row0","col0","row1","col1","area","count_cracked","percent_cracked"])
        for c in grid_stats:
            w.writerow([context.get("image_id",""),context.get("folder_main",""),context.get("folder_date",""),context.get("band",""),c["cell"],c["pix_thr"],c["row0"],c["col0"],c["row1"],c["col1"],c["area"],c["count_cracked"],c["percent_cracked"]])
    logger.info("CSV exported: %s (rows=%d)", fp, len(grid_stats))
    return fp

# ===== Pipeline =====

def run_patch_analysis(hdr_path: str, lda_path: str, scaler_path: Optional[str], band: int = 0, cell_size: int = 64, pix_thr: float = 0.5, out_dir: str = "grid_results") -> Tuple[str, str]:
    logger.info("Params -> band=%d cell=%d thr=%.2f", band, cell_size, pix_thr)
    lda, scaler, pos_idx = load_model_and_scaler(lda_path, scaler_path)
    cube = load_cube(hdr_path)
    prob_map = per_pixel_probs(cube, lda, scaler, pos_idx)
    grid_stats = analyze_grid(prob_map, cell_size, pix_thr)

    band_img = cv2.normalize(cube[:, :, band], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay_img = overlay_on_band(band_img, grid_stats)

    os.makedirs(out_dir, exist_ok=True)
    overlay_path = os.path.join(out_dir, "overlay.png")
    cv2.imwrite(overlay_path, overlay_img)
    logger.info("Overlay saved: %s", overlay_path)

    folder_main = os.path.basename(os.path.dirname(hdr_path))
    folder_date = os.path.basename(os.path.dirname(os.path.dirname(hdr_path)))
    ctx = {"image_id": os.path.splitext(os.path.basename(hdr_path))[0], "folder_main": folder_main, "folder_date": folder_date, "band": band, "cell": cell_size, "pix_thr": pix_thr}
    csv_path = export_grid_csv(grid_stats, ctx, out_dir)
    return overlay_path, csv_path

# ===== Threaded worker =====
class AnalysisThread(QThread):
    finished_signal = pyqtSignal(object, str)  # (overlay_rgb_array or None, csv_path)
    def __init__(self, hdr_path, lda_path, scaler_path, band, cell_size, pix_thr, out_dir):
        super().__init__()
        self.hdr_path, self.lda_path, self.scaler_path = hdr_path, lda_path, scaler_path
        self.band, self.cell_size, self.pix_thr, self.out_dir = band, cell_size, pix_thr, out_dir
    def run(self):
        try:
            t0 = time.perf_counter()
            overlay_path, csv_path = run_patch_analysis(self.hdr_path, self.lda_path, self.scaler_path, self.band, self.cell_size, self.pix_thr, self.out_dir)
            bgr = cv2.imread(overlay_path)
            if bgr is None:
                logger.warning("Overlay not found after analysis: %s", overlay_path)
                self.finished_signal.emit(None, csv_path); return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            logger.info("Analysis done in %.2fs", time.perf_counter() - t0)
            self.finished_signal.emit(rgb, csv_path)
        except Exception as e:
            logger.exception("AnalysisThread failed: %s", e)
            self.finished_signal.emit(None, "")

# ===== UI =====
class HSILDAViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI LDA Patch Classifier (All-in-One + UI Legend)")
        self.setGeometry(100, 100, 1250, 640)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)

        self.rgb_image: Optional[np.ndarray] = None
        self.hsi_cube: Optional[np.ndarray] = None
        self.current_band: int = 0
        self.folder_path: str = ""
        self.hdr_path: Optional[str] = None

        self.lda_path: Optional[str] = DEFAULT_LDA_PATH if os.path.exists(DEFAULT_LDA_PATH) else None
        self.scaler_path: Optional[str] = None

        # runtime params
        self.cell_size = 64
        self.pix_thr = 0.5

        self._build_ui()
        self._auto_load_model()
        self._refresh_legend()

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        L = QVBoxLayout(c)

        # --- Top row: images + legend on the right ---
        imgs = QHBoxLayout(); L.addLayout(imgs)
        self.rgb_label = QLabel("RGB"); self.rgb_label.setFixedSize(512, 512); self.rgb_label.setAlignment(Qt.AlignCenter); imgs.addWidget(self.rgb_label)
        self.hsi_label = QLabel("HSI Band"); self.hsi_label.setFixedSize(512, 512); self.hsi_label.setAlignment(Qt.AlignCenter); imgs.addWidget(self.hsi_label)
        self.legend_label = QLabel("Legend"); self.legend_label.setFixedSize(240, 140); self.legend_label.setAlignment(Qt.AlignTop | Qt.AlignLeft); imgs.addWidget(self.legend_label)

        # --- Controls row ---
        controls = QHBoxLayout(); L.addLayout(controls)
        btn = QPushButton("Load Images"); btn.clicked.connect(self._choose_folder); controls.addWidget(btn)

        # Band slider
        self.band_slider = QSlider(Qt.Horizontal); self.band_slider.setMinimum(0); self.band_slider.setValue(0); self.band_slider.valueChanged.connect(self._update_band); controls.addWidget(QLabel("Band")); controls.addWidget(self.band_slider)

        # Cell size spin
        controls.addWidget(QLabel("Cell"))
        self.cell_spin = QSpinBox(); self.cell_spin.setRange(8, 256); self.cell_spin.setSingleStep(8); self.cell_spin.setValue(self.cell_size); self.cell_spin.valueChanged.connect(self._update_cell); controls.addWidget(self.cell_spin)

        # Pixel threshold spin
        controls.addWidget(QLabel("Prob thr"))
        self.thr_spin = QDoubleSpinBox(); self.thr_spin.setRange(0.0, 1.0); self.thr_spin.setSingleStep(0.05); self.thr_spin.setDecimals(2); self.thr_spin.setValue(self.pix_thr); self.thr_spin.valueChanged.connect(self._update_thr); controls.addWidget(self.thr_spin)

        # Grid size display (new)
        controls.addWidget(QLabel("Grid"))
        self.grid_size_label = QLabel("N/A"); self.grid_size_label.setFixedWidth(140); controls.addWidget(self.grid_size_label)

        # Threshold buckets editor
        controls.addWidget(QLabel("Legend % (CSV)"))
        self.bucket_edit = QLineEdit("10,20,30,40"); self.bucket_edit.setFixedWidth(120); controls.addWidget(self.bucket_edit)
        apply_btn = QPushButton("Apply legend"); apply_btn.clicked.connect(self._apply_legend_from_text); controls.addWidget(apply_btn)

        # Run button + model label
        run_btn = QPushButton("Run Patch Analysis"); run_btn.clicked.connect(self._run_patches); controls.addWidget(run_btn)
        self.models_label = QLabel(f"Model: {os.path.basename(self.lda_path) if self.lda_path else 'None'}"); controls.addWidget(self.models_label)

        self.status_bar.showMessage("Load a folder, set params, then Run Patch Analysis.")

    def _auto_load_model(self):
        if not self.lda_path:
            self.status_bar.showMessage("Default LDA not found. Set DEFAULT_LDA_PATH or pick another model."); return
        cand = find_scaler(os.path.dirname(self.lda_path), [os.path.basename(self.lda_path)])
        self.scaler_path = cand
        self.models_label.setText(f"Model: {os.path.basename(self.lda_path)}")

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", DEFAULT_SEARCH_FOLDER)
        if not folder:
            self.status_bar.showMessage("No folder selected."); return
        self.folder_path = folder
        self._load_images(folder)

    def _load_images(self, folder: str):
        hs = os.path.join(folder, "HS")
        rgb_files = [f for f in os.listdir(hs) if f.lower().endswith(".png")]
        if not rgb_files: raise FileNotFoundError("No RGB image in HS folder")
        rgb = cv2.cvtColor(cv2.imread(os.path.join(hs, rgb_files[0])), cv2.COLOR_BGR2RGB)
        self._show_image(rgb, self.rgb_label)

        res = os.path.join(hs, "results")
        hdr_files = [f for f in os.listdir(res) if f.lower().endswith(".hdr")]
        if not hdr_files: raise FileNotFoundError("No .hdr in HS/results")
        self.hdr_path = os.path.join(res, hdr_files[0])
        self.hsi_cube = load_cube(self.hdr_path)
        self.band_slider.setMaximum(self.hsi_cube.shape[2] - 1)
        self._update_band()
        # update grid label when cube is loaded
        self._update_grid_label()
        self.status_bar.showMessage(f"Loaded images from {folder}")

    def _update_band(self):
        if self.hsi_cube is None: return
        self.current_band = self.band_slider.value()
        band = cv2.normalize(self.hsi_cube[:, :, self.current_band], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        band = cv2.rotate(band, cv2.ROTATE_90_CLOCKWISE)
        self._show_image(band, self.hsi_label, True)
        self.status_bar.showMessage(f"Band {self.current_band}")

    def _update_cell(self, v: int):
        self.cell_size = int(v)
        # update and log grid size
        self._update_grid_label()
        logger.info("Cell size changed -> %d", self.cell_size)
        self.status_bar.showMessage(f"Cell size set to {self.cell_size}")

    def _update_thr(self, v: float):
        self.pix_thr = float(v)

    def _show_image(self, img: np.ndarray, label: QLabel, is_hsi: bool = False):
        if is_hsi and img.ndim == 2:
            h, w = img.shape; q = QImage(img.data, w, h, QImage.Format_Grayscale8)
        else:
            h, w, ch = img.shape; q = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q)); label.setScaledContents(True)

    def _refresh_legend(self):
        img = render_legend_image(GRID_COLOR_BUCKETS)
        h, w, _ = img.shape
        q = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        self.legend_label.setPixmap(QPixmap.fromImage(q))
        self.legend_label.setScaledContents(True)

    def _apply_legend_from_text(self):
        global GRID_COLOR_BUCKETS
        text = self.bucket_edit.text().strip()
        try:
            # parse and sort ascending like [5,10,20,30,40]
            vals = sorted({int(x) for x in text.split(',') if x.strip() != ''})
            if not vals:
                raise ValueError
            # yellow to red gradient colors (BGR format)
            gradient_colors = [
                (0, 255, 255),    # yellow
                (0, 215, 255),    # yellow-orange
                (0, 140, 255),    # orange
                (0, 69, 255),     # red-orange
                (0, 0, 255),      # red
            ]
            # map increasing thresholds to increasing color intensity (yellow->red)
            vals = vals[:len(gradient_colors)]
            GRID_COLOR_BUCKETS = []
            for i, thr in enumerate(reversed(vals)):  # highest threshold first
                color_idx = len(vals) - 1 - (len(vals) - 1 - i)  # map to color intensity
                GRID_COLOR_BUCKETS.append((thr, gradient_colors[min(color_idx, len(gradient_colors)-1)]))

            logger.info("New yellow-to-red legend thresholds: %s", GRID_COLOR_BUCKETS)
            self._refresh_legend()
        except Exception:
            self.status_bar.showMessage("Invalid legend CSV. Use numbers like 5,10,20,30,40")

    def _run_patches(self):
        if not self.hdr_path:
            self.status_bar.showMessage("Load an HSI folder first."); return
        if not self.lda_path:
            self.status_bar.showMessage("No LDA model configured."); return
        band = self.current_band; cell = self.cell_size; thr = self.pix_thr
        # log grid rows/cols before running
        if self.hsi_cube is not None:
            H, W, _ = self.hsi_cube.shape
            rows = (H + cell - 1) // cell
            cols = (W + cell - 1) // cell
            logger.info("Running analysis -> band=%d cell=%d thr=%.2f | grid=%dx%d", band, cell, thr, rows, cols)
        model_name = os.path.splitext(os.path.basename(self.lda_path))[0]
        out_dir = os.path.join(os.path.dirname(self.hdr_path), f"grid_results_{model_name}")
        os.makedirs(out_dir, exist_ok=True)
        self.worker = AnalysisThread(self.hdr_path, self.lda_path, self.scaler_path, band, cell, thr, out_dir)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start(); self.status_bar.showMessage("Running analysis…")

    # New helper: update grid size label
    def _update_grid_label(self):
        if self.hsi_cube is None:
            self.grid_size_label.setText("N/A")
            return
        H, W, _ = self.hsi_cube.shape
        cell = max(1, int(self.cell_size))
        rows = (H + cell - 1) // cell
        cols = (W + cell - 1) // cell
        self.grid_size_label.setText(f"{rows} x {cols} (cell={cell})")
        logger.info("Grid size updated: rows=%d cols=%d cell=%d", rows, cols, cell)

    def _on_finished(self, overlay_rgb: Optional[np.ndarray], csv_path: str):
        if overlay_rgb is not None:
            self._show_image(overlay_rgb, self.hsi_label, True)
            self.status_bar.showMessage(f"Done. CSV: {os.path.basename(csv_path)}")
        else:
            self.status_bar.showMessage("Analysis finished (no overlay).")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = HSILDAViewer(); w.show()
    sys.exit(app.exec_())