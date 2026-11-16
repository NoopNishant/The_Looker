"""
core/preprocessing.py - OpenCV Preprocessing Pipeline.

Features:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for topology & micro-relief enhancement.
2. Screen-Flash Differencing Module for active optical watermark & security thread extraction.
"""

from typing import Dict, Optional, Tuple, Union
import time
import numpy as np
import cv2


class CLAHEEnhancer:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) processor.
    Enhances local micro-contrast without blowing out highlights or over-amplifying background noise.
    Operates in LAB color space to preserve chromatic fidelity.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size
        )

    def enhance_bgr(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Enhance local contrast of a BGR image by applying CLAHE to the Luminance (L) channel.
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image is empty or None")

        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def enhance_gray(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Enhance local contrast of a grayscale image.
        """
        if gray_image is None or gray_image.size == 0:
            raise ValueError("Input image is empty or None")

        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray_image)

    def extract_topological_gradients(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Compute high-frequency topological relief map (Sobel gradient magnitude on CLAHE L-channel)
        to inspect intaglio printing depth and paper grain.
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)

        grad_x = cv2.Sobel(l_enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_enhanced, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
        return grad_norm.astype(np.uint8)


class ScreenFlashDifferencing:
    """
    Active Optical Screen-Flash Differencing module.
    
    Operating Principle:
    A banknote is held in front of the webcam. Ambient light is recorded (I_ambient).
    The laptop monitor rapidly flashes a full-screen high-brightness white frame (I_flash).
    The differential transmission delta frame Delta_I = |I_flash - I_ambient| eliminates
    ambient background and reveals optical transmission features:
    - Embedded watermarks (Gandhi/Washington portrait, numeral electrotypes)
    - Embedded security thread magnetic windows
    - See-through registration patterns
    """

    def __init__(
        self,
        flash_window_name: str = "TheLooker_Flash_Trigger",
        flash_duration_ms: int = 180,
        threshold_ratio: float = 0.25
    ) -> None:
        self.flash_window_name = flash_window_name
        self.flash_duration_ms = flash_duration_ms
        self.threshold_ratio = threshold_ratio
        self.clahe = CLAHEEnhancer(clip_limit=3.0, tile_grid_size=(8, 8))

    def compute_difference(
        self,
        ambient_frame: np.ndarray,
        flash_frame: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute optical difference and extract watermark features from paired frames.
        
        Args:
            ambient_frame: Frame captured prior to screen flash.
            flash_frame: Frame captured during screen flash illumination.
        Returns:
            Dictionary with delta_bgr, delta_gray, watermark_mask, and watermark_enhanced.
        """
        if ambient_frame is None or flash_frame is None:
            raise ValueError("Frames must not be None")
        if ambient_frame.shape != flash_frame.shape:
            flash_frame = cv2.resize(flash_frame, (ambient_frame.shape[1], ambient_frame.shape[0]))

        # Absolute difference across color channels
        delta_bgr = cv2.absdiff(flash_frame, ambient_frame)

        # Convert to grayscale for transmission analysis
        delta_gray = cv2.cvtColor(delta_bgr, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to amplify faint watermark contours in delta image
        delta_enhanced = self.clahe.enhance_gray(delta_gray)

        # Gaussian smoothing to suppress sensor CMOS thermal noise
        blurred = cv2.GaussianBlur(delta_enhanced, (5, 5), 0)

        # Adaptive & Otsu multi-level thresholding for watermark boundary extraction
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening and closing to connect watermark fibers and eliminate stray speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Quantitative watermark presence metric (percentage of high-contrast transmission pixels)
        transmission_density = float(np.mean(cleaned_mask > 0))

        return {
            "delta_bgr": delta_bgr,
            "delta_gray": delta_gray,
            "delta_enhanced": delta_enhanced,
            "watermark_mask": cleaned_mask,
            "transmission_density": transmission_density
        }

    def capture_with_flash(
        self,
        cap: cv2.VideoCapture,
        screen_width: int = 1280,
        screen_height: int = 720
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Active acquisition procedure:
        1. Capture ambient frame.
        2. Create and display full-screen pure white canvas.
        3. Capture flash-illuminated frame.
        4. Close flash window.
        5. Output delta frame and extracted watermark mask.
        """
        # Step 1: Capture ambient frame
        ret, ambient_frame = cap.read()
        if not ret or ambient_frame is None:
            raise RuntimeError("Failed to capture ambient frame from video capture device")

        # Step 2: Trigger full-screen white canvas flash
        white_canvas = np.full((screen_height, screen_width, 3), 255, dtype=np.uint8)
        cv2.namedWindow(self.flash_window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.flash_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.flash_window_name, white_canvas)
        cv2.waitKey(max(1, self.flash_duration_ms // 2))

        # Flush any latent buffered frame
        cap.grab()

        # Step 3: Capture illuminated frame
        ret2, flash_frame = cap.read()

        # Step 4: Dismiss flash window
        cv2.destroyWindow(self.flash_window_name)

        if not ret2 or flash_frame is None:
            raise RuntimeError("Failed to capture flash frame during illumination")

        # Step 5: Compute optical difference
        diff_results = self.compute_difference(ambient_frame, flash_frame)
        diff_results["ambient_frame"] = ambient_frame
        diff_results["flash_frame"] = flash_frame

        return diff_results

    @staticmethod
    def simulate_screen_flash(
        base_frame: np.ndarray,
        watermark_type: str = "portrait"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize realistic ambient and flash frames for automated testing & headless verification.
        Emulates optical transmission of watermarked paper under back/front-lighting.
        """
        ambient = base_frame.copy()
        flash = base_frame.copy().astype(np.float32)

        h, w = base_frame.shape[:2]

        # Add uniform front-light reflection gain
        flash = flash * 1.25 + 25.0

        # Create simulated translucent watermark patch in designated region (e.g. left oval window)
        wm_center = (int(w * 0.3), int(h * 0.5))
        wm_axes = (int(w * 0.12), int(h * 0.22))

        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, wm_center, wm_axes, 0, 0, 360, 1.0, -1)
        # Security thread line
        thread_x = int(w * 0.55)
        cv2.line(mask, (thread_x, 0), (thread_x, h), 0.9, thickness=max(2, int(w * 0.01)))

        # Under screen flash, embedded watermark and thread modulate transmission intensity
        flash[:, :, 0] += mask * 45.0
        flash[:, :, 1] += mask * 50.0
        flash[:, :, 2] += mask * 40.0

        flash = np.clip(flash, 0, 255).astype(np.uint8)
        return ambient, flash

