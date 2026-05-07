import sys
import cv2
import numpy as np
import time
from scipy import stats
from scipy.ndimage import generic_filter
import streamlit as st
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------------
# IMAGE PROCESSING FUNCTIONS 
def ensure_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ensure_color(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
# ------------------------------------------------------------------------------------------
# 1. POINT OPERATIONS
def add_operation(img, value=50):
    gray = ensure_gray(img)
    return cv2.add(gray, np.full(gray.shape, value, dtype=np.uint8))

def subtract_operation(img, value=50):
    gray = ensure_gray(img)
    return cv2.subtract(gray, np.full(gray.shape, value, dtype=np.uint8))

def divide_operation(img, value=2):
    gray = ensure_gray(img)
    if value == 0: value = 1
    return cv2.divide(gray, np.full(gray.shape, value, dtype=np.uint8))

def complement_operation(img):
    return cv2.bitwise_not(ensure_gray(img))
# ---------------------------------------------------------------------------------------
# 2. COLOR IMAGE OPERATIONS
def change_red(img, value=50):
    res = ensure_color(img).copy()
    res[:, :, 2] = cv2.add(res[:, :, 2], value)
    return res

def swap_rg(img):
    res = ensure_color(img).copy()
    res[:, :, [1, 2]] = res[:, :, [2, 1]]
    return res

def remove_red(img):
    res = ensure_color(img).copy()
    res[:, :, 2] = 0
    return res
# -------------------------------------------------------------------------------------------
# 3. HISTOGRAM
def histogram_stretch(img):
    gray = ensure_gray(img)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

def histogram_equalization(img):
    gray = ensure_gray(img)
    return cv2.equalizeHist(gray)
# -------------------------------------------------------------------------------------------
# 4. NEIGHBORHOOD PROCESSING
def average_filter(img, k=3):
    return cv2.blur(img, (k, k))

def laplacian_filter(img):
    gray = ensure_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)

def max_filter(img, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(img, kernel)

def min_filter(img, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel)

def median_filter(img, k=3):
    return cv2.medianBlur(img, k)

def mode_filter(img, k=3):
    gray = ensure_gray(img)
    def get_mode(x):
        return stats.mode(x, keepdims=False)[0]
    return generic_filter(gray, get_mode, size=k)
# -------------------------------------------------------------------------------------------
# 5. SEGMENTATION
def basic_global_threshold(img, thresh=127):
    gray = ensure_gray(img)
    _, res = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return res

def automatic_threshold(img):
    gray = ensure_gray(img)
    _, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res

def adaptive_threshold(img, block_size=11, c=2):
    gray = ensure_gray(img)
    if block_size % 2 == 0: block_size += 1
    if block_size < 3: block_size = 3
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, c)
# -------------------------------------------------------------------------------------------
# 6. EDGE DETECTION
def sobel_detector(img):
    gray = ensure_gray(img)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# -------------------------------------------------------------------------------------------
# 7. MORPHOLOGY
def _get_binary(img):
    return automatic_threshold(img)

def dilation_op(img, k=3):
    binary = _get_binary(img)
    kernel = np.ones((k,k), np.uint8)
    return cv2.dilate(binary, kernel)

def erosion_op(img, k=3):
    binary = _get_binary(img)
    kernel = np.ones((k,k), np.uint8)
    return cv2.erode(binary, kernel)

def opening_op(img, k=3):
    binary = _get_binary(img)
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def internal_boundary(img, k=3):
    binary = _get_binary(img)
    return cv2.subtract(binary, erosion_op(img, k))

def external_boundary(img, k=3):
    binary = _get_binary(img)
    return cv2.subtract(dilation_op(img, k), binary)

def morphological_gradient(img, k=3):
    binary = _get_binary(img)
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
# -------------------------------------------------------------------------------------------
# 8. RESTORATION
def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    img_copy = img.copy()
    total_pixels = img.size if len(img.shape) == 2 else img.shape[0] * img.shape[1]
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    if len(img.shape) == 2:
        img_copy[salt_coords[0], salt_coords[1]] = 255
    else:
        img_copy[salt_coords[0], salt_coords[1], :] = 255
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
    if len(img.shape) == 2:
        img_copy[pepper_coords[0], pepper_coords[1]] = 0
    else:
        img_copy[pepper_coords[0], pepper_coords[1], :] = 0
    return img_copy

def add_gaussian_noise(img, mean=0, sigma=25):
    row, col = img.shape[:2]
    if len(img.shape) == 2:
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy = img + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        gauss = np.random.normal(mean, sigma, (row, col, 3))
        noisy = img + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def outlier_method_filter(img, k=3, threshold=30):
    gray = ensure_gray(img)
    median = cv2.medianBlur(gray, k)
    diff = cv2.absdiff(gray, median)
    mask = diff > threshold
    return np.where(mask, median, gray)

def image_averaging(img, num_frames=10):
    noisy_frames = []
    for _ in range(num_frames):
        noisy = add_gaussian_noise(img, sigma=25)
        noisy_frames.append(noisy.astype(np.float32))
    avg = np.mean(noisy_frames, axis=0).astype(np.uint8)
    return avg

# -----------------------------------------------------------------------------------
# IMAGE PROCESSOR STATE
class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.history = []

    def load_image(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Could not read image file.")
        self.load_image_from_mem(img)

    def load_image_from_mem(self, img):
        self.original_image = img.copy()
        self.current_image = img.copy()
        self.history = []

    def reset(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.history = []

    def clear(self):
        self.original_image = None
        self.current_image = None
        self.history = []

    def push_history(self):
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            if len(self.history) > 20: 
                self.history.pop(0)

    def undo(self):
        if self.history:
            self.current_image = self.history.pop()
            return True
        return False

    def apply_operation(self, op_code, params):
        if self.current_image is None: return
        self.push_history()
        img = self.current_image
        try:
            if op_code == 'add': self.current_image = add_operation(img, params.get('value', 50))
            elif op_code == 'subtract': self.current_image = subtract_operation(img, params.get('value', 50))
            elif op_code == 'divide': self.current_image = divide_operation(img, params.get('value', 2))
            elif op_code == 'complement': self.current_image = complement_operation(img)
            elif op_code == 'change_red': self.current_image = change_red(img, params.get('value', 50))
            elif op_code == 'swap_rg': self.current_image = swap_rg(img)
            elif op_code == 'remove_red': self.current_image = remove_red(img)
            elif op_code == 'hist_stretch': self.current_image = histogram_stretch(img)
            elif op_code == 'hist_equal': self.current_image = histogram_equalization(img)
            elif op_code == 'avg_filter': self.current_image = average_filter(img, params.get('k', 3))
            elif op_code == 'median_filter': self.current_image = median_filter(img, params.get('k', 3))
            elif op_code == 'max_filter': self.current_image = max_filter(img, params.get('k', 3))
            elif op_code == 'min_filter': self.current_image = min_filter(img, params.get('k', 3))
            elif op_code == 'mode_filter': self.current_image = mode_filter(img, params.get('k', 3))
            elif op_code == 'laplacian': self.current_image = laplacian_filter(img)
            elif op_code == 'basic_thresh': self.current_image = basic_global_threshold(img, params.get('value', 127))
            elif op_code == 'auto_thresh': self.current_image = automatic_threshold(img)
            elif op_code == 'adaptive_thresh': self.current_image = adaptive_threshold(img, params.get('k', 11))
            elif op_code == 'sobel': self.current_image = sobel_detector(img)
            elif op_code == 'dilation': self.current_image = dilation_op(img, params.get('k', 3))
            elif op_code == 'erosion': self.current_image = erosion_op(img, params.get('k', 3))
            elif op_code == 'opening': self.current_image = opening_op(img, params.get('k', 3))
            elif op_code == 'internal_boundary': self.current_image = internal_boundary(img, params.get('k', 3))
            elif op_code == 'external_boundary': self.current_image = external_boundary(img, params.get('k', 3))
            elif op_code == 'morph_gradient': self.current_image = morphological_gradient(img, params.get('k', 3))
            elif op_code == 'add_salt_pepper': self.current_image = add_salt_pepper_noise(img, params.get('salt_prob', 0.01), params.get('pepper_prob', 0.01))
            elif op_code == 'add_gaussian': self.current_image = add_gaussian_noise(img, params.get('mean', 0), params.get('sigma', 25))
            elif op_code == 'outlier_filter': self.current_image = outlier_method_filter(img, params.get('k', 3), params.get('threshold', 30))
            elif op_code == 'image_averaging': self.current_image = image_averaging(img, params.get('num_frames', 10))
            return True
        except Exception as e:
            self.undo() 
            raise e

# ------------------------------------------------------------------------------------------
# GUI APPLICATION (Streamlit with Matplotlib Subplots)

def main():
    st.set_page_config(page_title="Image Processing Studio", layout="wide", page_icon="❖$")
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: rgba(30, 41, 59, 0.7) !important; backdrop-filter: blur(12px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
        .stButton>button { width: 100%; border-radius: 12px; height: 3.2em; background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); color: white; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; transition: all 0.4s ease; }
        .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(99, 102, 241, 0.4); }
        .stDownloadButton>button { background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%); }
        h1, h2, h3 { font-family: 'Outfit', sans-serif; background: -webkit-linear-gradient(#fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stMetric { background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); }
        </style>
    """, unsafe_allow_html=True)

    if 'processor' not in st.session_state: st.session_state.processor = ImageProcessor()
    if 'last_op' not in st.session_state: st.session_state.last_op = None
    if 'exec_time' not in st.session_state: st.session_state.exec_time = 0

    with st.sidebar:
        st.title("❖ Studio Controls")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if st.session_state.processor.original_image is None or not np.array_equal(st.session_state.processor.original_image, img):
                st.session_state.processor.load_image_from_mem(img)
                st.session_state.last_op = "Load Image"

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↶ Undo") and st.session_state.processor.undo(): st.rerun()
        with col2:
            if st.button("⟳ Reset"): st.session_state.processor.reset(); st.session_state.last_op = "Reset"; st.rerun()
        if st.button("🗑 Clear Workspace"): st.session_state.processor.clear(); st.session_state.last_op = None; st.rerun()

        st.divider()
        category = st.selectbox("Category", ["Point Operations", "Color Operations", "Histogram", "Filters", "Segmentation", "Edge Detection", "Morphology", "Image Restoration"])
        op_map = {
            "Point Operations": [("Addition", "add"), ("Subtraction", "subtract"), ("Division", "divide"), ("Complement", "complement")],
            "Color Operations": [("Change Red", "change_red"), ("Swap R to G", "swap_rg"), ("Eliminate Red", "remove_red")],
            "Histogram": [("Stretching", "hist_stretch"), ("Equalization", "hist_equal")],
            "Filters": [("Average", "avg_filter"), ("Median", "median_filter"), ("Maximum", "max_filter"), ("Minimum", "min_filter"), ("Mode", "mode_filter"), ("Laplacian", "laplacian")],
            "Segmentation": [("Global Threshold", "basic_thresh"), ("Automatic", "auto_thresh"), ("Adaptive", "adaptive_thresh")],
            "Edge Detection": [("Sobel", "sobel")],
            "Morphology": [("Dilation", "dilation"), ("Erosion", "erosion"), ("Opening", "opening"), ("Internal Boundary", "internal_boundary"), ("External Boundary", "external_boundary"), ("Gradient", "morph_gradient")],
            "Image Restoration": [("Salt & Pepper", "add_salt_pepper"), ("Gaussian Noise", "add_gaussian"), ("Outlier Filter", "outlier_filter"), ("Averaging", "image_averaging")]
        }
        op_display_names = [n for n, c in op_map[category]]
        selected_op_name = st.selectbox("Operation", op_display_names)
        selected_op_code = next(c for n, c in op_map[category] if n == selected_op_name)

        params = {}
        if selected_op_code in ["add", "subtract", "divide", "change_red", "basic_thresh"]: params['value'] = st.slider("Value", 0, 255, 50 if selected_op_code != "divide" else 2)
        if selected_op_code in ["avg_filter", "median_filter", "max_filter", "min_filter", "mode_filter", "adaptive_thresh", "dilation", "erosion", "opening", "internal_boundary", "external_boundary", "morph_gradient", "outlier_filter"]:
            params['k'] = st.number_input("Kernel Size", 3, 31, 3, 2); 
            if params['k'] % 2 == 0: params['k'] += 1
        if selected_op_code == "add_salt_pepper": params['salt_prob'] = st.slider("Salt Prob", 0.0, 0.2, 0.01); params['pepper_prob'] = st.slider("Pepper Prob", 0.0, 0.2, 0.01)
        if selected_op_code == "add_gaussian": params['mean'] = st.slider("Mean", -50, 50, 0); params['sigma'] = st.slider("Sigma", 1, 100, 25)
        if selected_op_code == "outlier_filter": params['threshold'] = st.slider("Threshold", 1, 255, 30)
        if selected_op_code == "image_averaging": params['num_frames'] = st.slider("Frames", 2, 50, 10)

        if st.button("🚀 Apply Operation"):
            if st.session_state.processor.current_image is not None:
                start = time.time()
                try:
                    st.session_state.processor.apply_operation(selected_op_code, params)
                    st.session_state.last_op = selected_op_name
                    st.session_state.exec_time = (time.time() - start) * 1000
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")
            else: st.warning("Upload an image first")

    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>❖ Image Processing Studio</h1>", unsafe_allow_html=True)
    if st.session_state.processor.current_image is not None:
        p_img = st.session_state.processor.current_image
        h, w = p_img.shape[:2]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resolution", f"{w}x{h}")
        m2.metric("Last Operation", st.session_state.last_op or "None")
        m3.metric("Latency", f"{st.session_state.exec_time:.1f}ms")
        m4.metric("History", len(st.session_state.processor.history))
        st.divider()

        # --- Display with Matplotlib Subplot ---
        orig_rgb = cv2.cvtColor(st.session_state.processor.original_image, cv2.COLOR_BGR2RGB)
        if len(p_img.shape) == 3:
            proc_rgb = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
        else:
            proc_rgb = p_img  # grayscale

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original Image", fontsize=14, color='white')
        axes[0].axis('off')
        axes[1].imshow(proc_rgb, cmap='gray' if len(p_img.shape)==2 else None)
        axes[1].set_title("Processed Result", fontsize=14, color='white')
        axes[1].axis('off')
        fig.patch.set_facecolor('#0f172a')
        st.pyplot(fig)
        plt.close(fig)

        # Download button
        success, buff = cv2.imencode(".png", p_img)
        if success:
            st.download_button("💾 Download Result", buff.tobytes(), "processed.png", "image/png")
    else:
        st.info("👋 Welcome! Please upload an image to start processing.")
        # Placeholder image
        st.image("https://images.unsplash.com/photo-1541963463532-d68292c34b19?auto=format&fit=crop&w=1350&q=80", use_column_width=True)

if __name__ == "__main__":
    if "streamlit" in sys.modules: main()
    else:
        import subprocess
        try:
            import streamlit
            subprocess.run(["streamlit", "run", __file__])
        except ImportError: print("Streamlit not found. Install it with: pip install streamlit")