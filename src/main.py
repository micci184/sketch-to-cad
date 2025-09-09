#!/usr/bin/env python3
"""
Sketch-to-CAD Main Program
Convert hand-drawn CAD markups to DXF - GPT-5 Version

Usage:
    python src/main.py input/drawing.png
    python src/main.py input/drawing.jpg output/result.dxf
"""

import os
import sys
import cv2
import numpy as np
import ezdxf
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# .env の読み込み（OPENAI_API_KEY など）
load_dotenv()


# ===========================================
# Data Classes
# ===========================================

@dataclass
class CADElement:
    """Data class representing a CAD element"""
    element_type: str  # 'line', 'text', 'symbol', 'deletion'
    coordinates: List[Tuple[float, float]]
    content: Optional[str] = None
    color: str = 'black'
    layer: str = '0_EXISTING'
    confidence: float = 1.0


class AIApiError(Exception):
    """Custom exception for AI API related errors."""
    pass


@dataclass
class ProcessingResult:
    """Data class representing processing result"""
    success: bool
    output_path: Optional[str] = None
    element_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ===========================================
# Main Class
# ===========================================

class SketchToCAD:
    """Main class for converting hand-drawn markups to DXF"""
    
    # DXF layer definitions
    LAYERS = {
        '0_EXISTING': {'color': 7, 'desc': 'Existing CAD elements'},
        '1_ADDITION': {'color': 1, 'desc': 'Added elements (red pen)'},
        '2_DELETION': {'color': 6, 'desc': 'Deletion marks (X marks)'},
        '3_ANNOTATION': {'color': 2, 'desc': 'Notes and text'},
        '9_REVIEW': {'color': 4, 'desc': 'Elements requiring review'}
    }
    
    # HSV color detection ranges
    COLOR_RANGES = {
        'red': {
            'lower1': np.array([0, 50, 50]),
            'upper1': np.array([10, 255, 255]),
            'lower2': np.array([170, 50, 50]),
            'upper2': np.array([180, 255, 255])
        },
        'blue': {
            'lower': np.array([100, 50, 50]),
            'upper': np.array([130, 255, 255])
        }
    }
    
    def __init__(self, use_ai: bool = True, ai_provider: str = 'gpt5'):
        """
        Initialize
        
        Args:
            use_ai: Whether to use AI recognition
            ai_provider: AI provider ('gpt5', 'claude', 'gemini')
        """
        self.use_ai = use_ai
        self.ai_provider = ai_provider
        self.elements: List[CADElement] = []
        
        if use_ai:
            self._setup_ai()
    
    def _setup_ai(self):
        """Set up AI API"""
        # Use GPT-5 API as primary
        if self.ai_provider == 'gpt5':
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                logger.error("OPENAI_API_KEY not set")
                logger.error("Run: export OPENAI_API_KEY='your-key'")
                self.use_ai = False
            else:
                logger.info("AI key detected. Enabling GPT provider (gpt5 route)")
        else:
            # Other providers (for future extensions)
            api_keys = {
                'claude': os.getenv('CLAUDE_API_KEY'),
                'gemini': os.getenv('GOOGLE_AI_KEY')
            }
            self.api_key = api_keys.get(self.ai_provider)
            if not self.api_key:
                logger.warning(f"API key not set: {self.ai_provider}")
                self.use_ai = False
    
    # ===========================================
    # Image Preprocessing
    # ===========================================
    
    def preprocess_image(self, image_path: str) -> Dict:
        """
        Preprocess image
        
        Args:
            image_path: Input image path
            
        Returns:
            Preprocessed data
        """
        logger.info(f"Loading image: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        height, width = img.shape[:2]
        logger.info(f"  Image size: {width}x{height}")
        
        # iPhone image correction
        if self._is_iphone_image(img):
            logger.info("  iPhone image detected - applying corrections")
            img = self._enhance_iphone_image(img)
        
        # Color separation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Generate masks for each color
        red_mask = self._detect_color(hsv, 'red')
        blue_mask = self._detect_color(hsv, 'blue')
        black_mask = self._detect_black_elements(img)
        
        # Exclude black from red mask (avoid overlap)
        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(black_mask))
        
        return {
            'original': img,
            'red_mask': red_mask,
            'blue_mask': blue_mask,
            'black_mask': black_mask,
            'height': height,
            'width': width,
            'hsv': hsv
        }
    
    def _is_iphone_image(self, img: np.ndarray) -> bool:
        """Check if image is from iPhone"""
        height, width = img.shape[:2]
        # Check iPhone aspect ratios
        aspect_ratios = [16/9, 4/3, 19.5/9]  # iPhone standard ratios
        img_ratio = width / height
        
        for ratio in aspect_ratios:
            if abs(img_ratio - ratio) < 0.1 or abs(img_ratio - 1/ratio) < 0.1:
                return True
        return False
    
    def _enhance_iphone_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance iPhone image quality"""
        # Contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Noise reduction (light)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        return enhanced
    
    def _detect_color(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """Detect specified color"""
        ranges = self.COLOR_RANGES.get(color)
        if not ranges:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        if color == 'red':
            # Red needs two ranges
            mask1 = cv2.inRange(hsv, ranges['lower1'], ranges['upper1'])
            mask2 = cv2.inRange(hsv, ranges['lower2'], ranges['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
        
        # Morphological operations for noise removal and line connection
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        
        return mask
    
    def _detect_black_elements(self, img: np.ndarray) -> np.ndarray:
        """Detect black elements (existing CAD lines)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Enhance thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    # ===========================================
    # Element Detection
    # ===========================================
    
    def _save_debug_image(self, name: str, image: np.ndarray):
        """Saves an image to the debug directory if debug mode is enabled."""
        if os.getenv("SKETCH2CAD_DEBUG"):
            debug_dir = Path("output") / "debug"
            debug_dir.mkdir(exist_ok=True)
            # Use a consistent timestamp for the current run if available
            ts = getattr(self, "debug_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(str(debug_dir / f"{ts}_{name}.png"), image)

    def _crop_to_content(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crops the image to the bounding box of the main drawing area using morphological operations."""
        self._save_debug_image("01a_mask_before_morph", mask)

        # Use a large kernel for morphological operations to connect/remove large features
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Opening: Remove small noise and thin border connections
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self._save_debug_image("01b_mask_opened", opened_mask)

        # Closing: Connect disparate parts of the main drawing to form a solid blob
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        self._save_debug_image("01c_mask_closed", closed_mask)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours found after morphological operations. Skipping crop.")
            return img, mask
        # Find the largest contour, which should now be the main drawing area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        logger.info(f"Cropping to significant content area: x={x}, y={y}, w={w}, h={h}")
        cropped_img = img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w] # Use original mask for cropping to preserve all data within the ROI
        
        self._save_debug_image("00_cropped_input", cropped_img)
        return cropped_img, cropped_mask

    def _clean_black_mask(self, mask: np.ndarray) -> np.ndarray: 
        """Light cleanup for black mask (kept simple to avoid overfiltering)."""
        try:
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            return cleaned
        except Exception:
            return mask

    def detect_elements(self, processed_data: Dict) -> List[CADElement]:
        """Main element detection pipeline."""
        logger.info("Detecting elements...")
        self.debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        elements = []

        # --- Black Element Processing ---
        black_mask = processed_data['black_mask']
        self._save_debug_image("01_black_mask_initial", black_mask)

        # 1. Crop image to the main content area to remove external noise
        original_img = processed_data['original']
        cropped_img, cropped_black_mask = self._crop_to_content(original_img, black_mask)
        processed_data['original'] = cropped_img # Update for AI processing
        processed_data['height'], processed_data['width'] = cropped_img.shape[:2]

        # After cropping, all subsequent operations use the cropped mask
        self._save_debug_image("02_black_mask_cropped", cropped_black_mask)

        # 2. Detect and extract closed shapes (rectangles) from the cropped mask
        contours, _ = cv2.findContours(cropped_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles, rect_contours = self._detect_rectangles(contours, layer='0_EXISTING')
        elements.extend(rectangles)
        logger.info(f"  Detected rectangles: {len(rectangles)}")

        # 3. Remove rectangles from the mask to process lines separately
        line_mask = cropped_black_mask.copy()
        cv2.drawContours(line_mask, rect_contours, -1, (0), thickness=cv2.FILLED)
        self._save_debug_image("03_rectangles_removed", line_mask)

        # 4. Skeletonize the remaining mask to get clean centerlines
        skeleton = self._skeletonize(line_mask)
        self._save_debug_image("04_skeleton", skeleton)

        # 5. Detect lines from the skeletonized image
        lines = self._detect_lines_from_skeleton(
            skeleton, 
            'black', 
            '0_EXISTING',
            processed_data['original'].shape
        )
        elements.extend(lines)
        logger.info(f"  Detected lines from skeleton: {len(lines)}")

        # --- Red Element Processing ---
        red_lines = self._detect_lines(processed_data['red_mask'], 'red', '1_ADDITION')
        elements.extend(red_lines)
        logger.info(f"  Added lines (red): {len(red_lines)}")

        # --- Deletion Mark Processing ---
        deletions = self._detect_deletion_marks(processed_data['red_mask'])
        elements.extend(deletions)
        logger.info(f"  Deletion marks: {len(deletions)}")

        # --- Blue Element Processing ---
        if np.any(processed_data['blue_mask']):
            blue_lines = self._detect_lines(processed_data['blue_mask'], 'blue', '3_ANNOTATION')
            elements.extend(blue_lines)
            logger.info(f"  Supplementary lines (blue): {len(blue_lines)}")

        return elements

    def _correct_orthogonality(self, points: List[Tuple[float, float]], tolerance_deg: float = 5.0) -> List[Tuple[float, float]]:
        """Corrects near-horizontal and near-vertical lines to be perfectly orthogonal."""
        if len(points) < 2:
            return points
        
        corrected_points = list(points)

        for i in range(len(corrected_points) - 1):
            p1 = corrected_points[i]
            p2 = corrected_points[i+1]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            angle = np.degrees(np.arctan2(dy, dx))

            if abs(angle) < tolerance_deg or abs(abs(angle) - 180.0) < tolerance_deg:
                corrected_points[i+1] = (p2[0], p1[1])
            elif abs(abs(angle) - 90.0) < tolerance_deg:
                corrected_points[i+1] = (p1[0], p2[1])
                
        return corrected_points

    def _detect_rectangles(self, contours: List[np.ndarray], layer: str) -> Tuple[List[CADElement], List[np.ndarray]]:
        """Detects rectangles from a list of contours."""
        rectangles = []
        rect_contours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 400 and cv2.isContourConvex(approx):
                coords = [tuple(p[0]) for p in approx]
                corrected_coords = self._correct_orthogonality(coords)
                element = CADElement(element_type='polyline', coordinates=corrected_coords, layer=layer)
                rectangles.append(element)
                rect_contours.append(contour)
        return rectangles, rect_contours

    def _detect_lines_from_skeleton(self, skeleton: np.ndarray, color: str, layer: str, original_shape: Tuple[int, int]) -> List[CADElement]:
        """Detects lines from a skeletonized image, with border exclusion and robust post-processing."""
        # Exclude image border to avoid frame noise
        skel = self._exclude_border(skeleton, border=14)
        # Tuned Hough parameters
        lines = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=12)
        
        # --- Debug visualization of raw and merged lines ---
        if os.getenv("SKETCH2CAD_DEBUG"):
            debug_img_raw = np.zeros(original_shape, dtype=np.uint8)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_img_raw, (x1, y1), (x2, y2), (255, 0, 0), 1)
            self._save_debug_image("05_hough_lines_raw", debug_img_raw)
        if lines is None:
            return []

        merged_lines = self._merge_lines(lines)
        # Post-process: min length, angle snap, dedup
        merged_lines = self._postprocess_lines([np.array([l[0], l[1], l[2], l[3]]) for l in merged_lines])

        if os.getenv("SKETCH2CAD_DEBUG"):
            debug_img_merged = np.zeros(original_shape, dtype=np.uint8)
            for line in merged_lines:
                x1, y1, x2, y2 = line
                cv2.line(debug_img_merged, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            self._save_debug_image("06_merged_lines_final", debug_img_merged)
        
        elements = []
        for line_segment in merged_lines:
            p1 = (float(line_segment[0]), float(line_segment[1]))
            p2 = (float(line_segment[2]), float(line_segment[3]))
            corrected_points = self._correct_orthogonality([p1, p2])
            
            element = CADElement(
                element_type='line',
                coordinates=corrected_points,
                color=color,
                layer=layer
            )
            elements.append(element)
        return elements

    def _group_lines_by_angle(self, lines: np.ndarray, angle_thresh_deg: float) -> Dict[int, List[np.ndarray]]:
        """Groups lines by angle."""
        groups = {}
        angle_keys = {}
        for line in lines:
            angle = self._get_line_angle_deg(line[0])
            found_group = False
            # Check against existing group angles
            for key_angle in angle_keys:
                if abs(angle - key_angle) < angle_thresh_deg:
                    groups[key_angle].append(line[0])
                    found_group = True
                    break
            if not found_group:
                groups[angle] = [line[0]]
                angle_keys[angle] = True
        return groups

    def _merge_lines_in_group(self, group: List[np.ndarray], max_dist: float) -> List[np.ndarray]:
        """Merges lines within a single angle group."""
        if not group:
            return []
        
        # Sort lines by their starting point to handle them in order
        group.sort(key=lambda l: (l[0], l[1]))

        merged = []
        used = [False] * len(group)

        for i in range(len(group)):
            if used[i]:
                continue
            
            current_line = group[i]
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                
                other_line = group[j]
                if self._are_lines_mergeable(current_line, other_line, max_dist, angle_thresh_deg=10):
                    current_line = self._extend_line(current_line, other_line)
                    used[j] = True
            
            merged.append(current_line)
            used[i] = True

        return merged

    def _merge_lines(self, lines: np.ndarray, max_dist: float = 20.0, angle_thresh_deg: float = 5.0) -> List[np.ndarray]:
        """Merges collinear and close line segments into longer lines."""
        if lines is None or len(lines) < 2:
            return [] if lines is None else [l[0] for l in lines]

        line_groups = self._group_lines_by_angle(lines, angle_thresh_deg)
        
        all_merged_lines = []
        for angle_key in line_groups:
            merged_group = self._merge_lines_in_group(line_groups[angle_key], max_dist)
            all_merged_lines.extend(merged_group)

        return all_merged_lines

    def _get_line_angle_deg(self, line: np.ndarray) -> float:
        """Calculates the angle of a line in degrees."""
        return np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))

    # -------------------- Line post-process helpers --------------------
    def _exclude_border(self, img: np.ndarray, border: int = 12) -> np.ndarray:
        """Zero-out a border band to avoid frame noise."""
        h, w = img.shape[:2]
        out = img.copy()
        out[:border, :] = 0
        out[h-border:, :] = 0
        out[:, :border] = 0
        out[:, w-border:] = 0
        return out

    def _line_length(self, line: np.ndarray) -> float:
        return float(np.hypot(line[2] - line[0], line[3] - line[1]))

    def _angle_snap_line(self, line: np.ndarray, tol_deg: float = 5.0) -> np.ndarray:
        """Snap near-horizontal/vertical lines to perfect orthogonal."""
        x1, y1, x2, y2 = map(float, line)
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < tol_deg or abs(abs(angle) - 180.0) < tol_deg:
            # horizontal -> align y to average
            y = (y1 + y2) / 2.0
            return np.array([x1, y, x2, y])
        if abs(abs(angle) - 90.0) < tol_deg:
            # vertical -> align x to average
            x = (x1 + x2) / 2.0
            return np.array([x, y1, x, y2])
        return np.array([x1, y1, x2, y2])

    def _dedup_lines(self, lines: List[np.ndarray], grid: float = 2.0) -> List[np.ndarray]:
        """Remove near-duplicate lines by snapping to a coarse grid and using a set."""
        def snap(v: float) -> int:
            return int(round(v / grid))
        seen = set()
        out: List[np.ndarray] = []
        for ln in lines:
            x1, y1, x2, y2 = ln
            key = tuple(sorted([(snap(x1), snap(y1)), (snap(x2), snap(y2))]))
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)
        return out

    def _postprocess_lines(self, lines: List[np.ndarray], min_len: float = 30.0) -> List[np.ndarray]:
        """Apply min-length, angle snap, and deduplication."""
        # Min length
        filtered = [ln for ln in lines if self._line_length(ln) >= min_len]
        # Angle snap
        snapped = [self._angle_snap_line(ln) for ln in filtered]
        # Dedup
        return self._dedup_lines(snapped)

    def _are_lines_mergeable(self, line1: np.ndarray, line2: np.ndarray, max_dist: float, angle_thresh_deg: float) -> bool:
        """Checks if two lines can be merged."""
        angle1 = self._get_line_angle_deg(line1)
        angle2 = self._get_line_angle_deg(line2)
        if abs(angle1 - angle2) > angle_thresh_deg:
            return False

        # Check proximity of endpoints
        dist1 = np.linalg.norm(np.array(line1[2:4]) - np.array(line2[0:2]))
        dist2 = np.linalg.norm(np.array(line2[2:4]) - np.array(line1[0:2]))
        return min(dist1, dist2) < max_dist

    def _extend_line(self, line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
        """Extends line1 to encompass line2 by finding the two most distant endpoints."""
        points = np.array([line1[0:2], line1[2:4], line2[0:2], line2[2:4]])
        max_dist = 0
        p1, p2 = None, None
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    p1, p2 = points[i], points[j]
        return np.array([p1[0], p1[1], p2[0], p2[1]])

    def _detect_lines(self, mask: np.ndarray, color: str, layer: str) -> List[CADElement]:
        """Detects simple lines, primarily for non-black elements like annotations."""
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=25, minLineLength=25, maxLineGap=15)

        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                element = CADElement(
                    element_type='line',
                    coordinates=[(float(x1), float(y1)), (float(x2), float(y2))],
                    color=color,
                    layer=layer
                )
                detected.append(element)
        return detected

    def _detect_deletion_marks(self, mask: np.ndarray) -> List[CADElement]:

        """Detect deletion marks (X marks)"""
        deletions = []
        
        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 5000:  # Size filter
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check for square-like shape
            if 0.7 < aspect_ratio < 1.3:
                # Detect diagonal lines in ROI
                roi = mask[y:y+h, x:x+w]
                if self._has_cross_pattern(roi):
                    element = CADElement(
                        element_type='deletion',
                        coordinates=[(float(x+w/2), float(y+h/2))],
                        layer='2_DELETION'
                    )
                    deletions.append(element)
        
        return deletions

    def _skeletonize(self, img: np.ndarray) -> np.ndarray:
        """Reduces a binary image to a 1-pixel-wide skeleton."""
        skeleton = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        img = img.copy() # Work on a copy

        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skeleton

    def _has_cross_pattern(self, roi: np.ndarray) -> bool:
        """Detect X pattern"""
        if roi.size == 0:
            return False
        
        # Detect diagonal lines
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 15, minLineLength=10, maxLineGap=5)
        
        if lines is not None and len(lines) >= 2:
            angles = []
            for line in lines[:5]:  # Check first 5 lines only
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                angles.append(abs(angle))
            
            # Check for lines near 45 and 135 degrees
            has_diagonal1 = any(30 < a < 60 for a in angles)
            has_diagonal2 = any(120 < a < 150 for a in angles)
            
            return has_diagonal1 and has_diagonal2
        
        return False
    
    # ===========================================
    # AI Recognition (GPT-5)
    # ===========================================
    
    async def recognize_with_ai(self, image_data: Dict) -> List[CADElement]:
        """
        AI-powered recognition
        
        Args:
            image_data: Image data
            
        Returns:
            AI recognition results
        """
        if not self.use_ai:
            return []
        
        logger.info(f"Running AI recognition ({self.ai_provider})...")
        
        if self.ai_provider == 'gpt5':
            return await self._call_gpt5_api(image_data)
        elif self.ai_provider == 'claude':
            return await self._call_claude_api(image_data)
        elif self.ai_provider == 'gemini':
            return await self._call_gemini_api(image_data)
        
        return []
    
    async def _call_gpt5_api(self, image_data: Dict) -> List[CADElement]:
        """GPT-5 API call (August 2025 latest)"""
        try:
            import openai
            import base64
            import os

            # Temporarily unset proxy env variables to fix httpx/openai init issue
            proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            original_proxies = {key: os.environ.pop(key, None) for key in proxy_keys}

            try:
                client = openai.OpenAI(api_key=self.api_key)
            finally:
                # Restore original proxy settings
                for key, value in original_proxies.items():
                    if value is not None:
                        os.environ[key] = value
            
            # Encode image to Base64
            _, buffer = cv2.imencode('.png', image_data['original'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Use widely available multimodal model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a CAD drawing analysis expert. Identify handwritten modifications in architectural drawings."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this architectural drawing. Identify all handwritten modifications, including text and simple geometric shapes like circles.
                                Return the results in a strict JSON format. The JSON object must have a key named 'elements' which is a list of objects.
                                Each object in the list must describe a single element and have a 'type' key.

                                Supported types are:
                                - 'text': For handwritten text.
                                - 'circle': For hand-drawn circles.

                                For 'text' elements, include these keys:
                                - 'type': "text"
                                - 'content': The recognized text (string).
                                - 'position': A list containing the [x, y] coordinates of the text's starting point.

                                For 'circle' elements, include these keys:
                                - 'type': "circle"
                                - 'center': A list containing the [x, y] coordinates of the circle's center.
                                - 'radius': The approximate radius of the circle (number).

                                Example output:
                                {
                                  "elements": [
                                    { "type": "text", "content": "トイレ", "position": [500, 600] },
                                    { "type": "circle", "center": [300, 400], "radius": 25 }
                                  ]
                                }
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.2  # Lower temperature for accuracy
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.info(f"  GPT-5 raw response: {content}")
            
            # Parse JSON and create CADElements
            import json
            import re
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                elements = self._parse_ai_response(json_match.group())
                logger.info(f"  GPT-5 recognized {len(elements)} elements")
                return elements
                
        except openai.RateLimitError as e:
            msg = f"OpenAI API rate limit or quota exceeded: {e}"
            logger.error(msg)
            raise AIApiError(msg) from e
        except Exception as e:
            import traceback
            msg = f"An unexpected GPT API error occurred: {e}\n{traceback.format_exc()}"
            logger.error(msg)
            raise AIApiError(msg) from e
    
    async def _call_claude_api(self, _image_data: Dict) -> List[CADElement]:
        """Claude API call (optional)"""
        # For future extension
        return []
    
    def _parse_ai_response(self, json_string: str) -> List[CADElement]:
        """Parses the JSON response from the AI API and converts it to CADElement objects."""
        import json
        data = json.loads(json_string)
        elements = []
        
        for item in data.get('elements', []):
            item_type = item.get('type')

            if item_type == 'text':
                pos = item.get('position')
                text_content = item.get('content')
                if pos and text_content and len(pos) == 2:
                    elements.append(CADElement(
                        element_type='text',
                        coordinates=[(pos[0], pos[1])],
                        content=text_content,
                        layer='3_ANNOTATION',
                        confidence=0.9
                    ))

            elif item_type == 'circle':
                center = item.get('center')
                radius = item.get('radius')
                if center and radius and len(center) == 2:
                    elements.append(CADElement(
                        element_type='circle',
                        coordinates=[(center[0], center[1])], # Store center
                        content=str(radius), # Store radius in content field
                        layer='1_ADDITION',
                        confidence=0.9
                    ))
        return elements

    async def _call_gemini_api(self, _image_data: Dict) -> List[CADElement]:
        """Gemini API call (optional)"""
        # For future extension
        return []
    
    # ===========================================
    # DXF Conversion
    # ===========================================
    
    def convert_to_dxf(self, elements: List[CADElement], output_path: str, image_height: int) -> bool:
        """
        Convert to DXF file
        
        Args:
            elements: Detected elements
            output_path: Output file path
            
        Returns:
            Success/Failure
        """
        logger.info("Starting DXF conversion...")
        
        try:
            # Create DXF document
            doc = ezdxf.new('R2018')

            # Add a text style for Japanese fonts to prevent font errors
            doc.styles.new('jp_font_style', dxfattribs={'font': 'BIZ UDGothic'})

            msp = doc.modelspace()
            
            # Create layers
            for layer_name, props in self.LAYERS.items():
                doc.layers.add(
                    name=layer_name,
                    color=props['color']
                )
            
            # Add elements
            added_count = 0
            for element in elements:
                if self._add_element_to_dxf(msp, element, image_height):
                    added_count += 1
            
            # Save file
            doc.saveas(output_path)
            logger.info(f"  DXF saved: {output_path}")
            logger.info(f"  Elements added: {added_count}/{len(elements)}")
            
            return True
            
        except Exception as e:
            logger.error(f"DXF conversion error: {e}")
            return False

    # ================================
    # Mask -> DXF (Vectorize contours)
    # ================================
    def export_mask_to_dxf(self, mask_path: str, output_path: Optional[str] = None) -> ProcessingResult:
        """Vectorize a binary mask PNG and export as DXF polylines.

        The white regions (value>0) are treated as filled shapes; outer contours are exported
        as closed polylines on layer '0_EXISTING'.
        """
        import time
        start = time.time()
        try:
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot load mask: {mask_path}")

            # Ensure binary
            _, bin_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Find outer contours
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            elements: List[CADElement] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:  # tiny noise
                    continue
                peri = cv2.arcLength(cnt, True)
                epsilon = max(1.0, 0.003 * peri)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                coords = [tuple(pt[0]) for pt in approx]
                if len(coords) >= 3:
                    elements.append(CADElement(
                        element_type='polyline',
                        coordinates=coords,
                        layer='0_EXISTING'
                    ))

            if output_path is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = Path(mask_path).stem
                output_path = f"output/{stem}_vectorized_{ts}.dxf"
                Path("output").mkdir(exist_ok=True)

            ok = self.convert_to_dxf(elements, output_path, image_height=bin_mask.shape[0])
            if ok:
                return ProcessingResult(
                    success=True,
                    output_path=output_path,
                    element_count=len(elements),
                    processing_time=time.time() - start,
                )
            else:
                return ProcessingResult(success=False, error_message="DXF conversion failed", processing_time=time.time() - start)
        except Exception as e:
            logger.error(f"Mask->DXF error: {e}")
            return ProcessingResult(success=False, error_message=str(e), processing_time=time.time() - start)
    
    def _add_line_to_dxf(self, msp, element: CADElement, image_height: int) -> bool:
        if len(element.coordinates) < 2: return False
        start_cad = self._pixel_to_cad(element.coordinates[0], image_height)
        end_cad = self._pixel_to_cad(element.coordinates[1], image_height)
        logger.info(f"Adding LINE: from {start_cad} to {end_cad}")
        msp.add_line(start=start_cad, end=end_cad, dxfattribs={'layer': element.layer})
        return True

    def _add_polyline_to_dxf(self, msp, element: CADElement, image_height: int) -> bool:
        if len(element.coordinates) < 2: return False
        cad_coords = [self._pixel_to_cad(c, image_height) for c in element.coordinates]
        logger.info(f"Adding POLYLINE with {len(cad_coords)} vertices, starting at {cad_coords[0]}")
        msp.add_lwpolyline(cad_coords, close=True, dxfattribs={'layer': element.layer})
        return True

    def _add_circle_to_dxf(self, msp, element: CADElement, image_height: int) -> bool:
        if not element.coordinates or not element.content: return False
        center_cad = self._pixel_to_cad(element.coordinates[0], image_height)
        radius = float(element.content)
        logger.info(f"Adding CIRCLE at {center_cad} with radius {radius}")
        msp.add_circle(center=center_cad, radius=radius, dxfattribs={'layer': element.layer})
        return True

    def _add_text_to_dxf(self, msp, element: CADElement, image_height: int) -> bool:
        if not element.coordinates or not element.content: return False
        cad_coord = self._pixel_to_cad(element.coordinates[0], image_height)
        logger.info(f"Adding TEXT: '{element.content}' at cad_coord={cad_coord}")
        msp.add_text(
            element.content,
            height=2.5,
            dxfattribs={'layer': element.layer, 'insert': cad_coord, 'style': 'jp_font_style'}
        )
        return True

    def _add_deletion_to_dxf(self, msp, element: CADElement, image_height: int) -> bool:
        if not element.coordinates: return False
        center_cad = self._pixel_to_cad(element.coordinates[0], image_height)
        size = 10.0  # mm
        logger.info(f"Adding DELETION mark at {center_cad}")
        msp.add_line(start=(center_cad[0]-size/2, center_cad[1]-size/2), end=(center_cad[0]+size/2, center_cad[1]+size/2), dxfattribs={'layer': element.layer, 'color': 1})
        msp.add_line(start=(center_cad[0]-size/2, center_cad[1]+size/2), end=(center_cad[0]+size/2, center_cad[1]-size/2), dxfattribs={'layer': element.layer, 'color': 1})
        return True

    def _add_element_to_dxf(self, msp, element: CADElement, image_height: int) -> bool:
        """Add element to DXF model space by dispatching to the correct handler."""
        handlers = {
            'line': self._add_line_to_dxf,
            'polyline': self._add_polyline_to_dxf,
            'circle': self._add_circle_to_dxf,
            'text': self._add_text_to_dxf,
            'deletion': self._add_deletion_to_dxf,
        }

        handler = handlers.get(element.element_type)
        if not handler:
            logger.warning(f"No DXF handler for element type: {element.element_type}")
            return False

        try:
            return handler(msp, element, image_height)
        except Exception as e:
            logger.error(f"Failed to add element {element.element_type} to DXF: {e}", exc_info=True)
            
        return False
    
    def _pixel_to_cad(self, coord: Tuple[float, float], image_height: int) -> Tuple[float, float]:
        """Convert pixel coordinates to CAD coordinates (Y-axis inverted)."""
        x = coord[0]
        y = image_height - coord[1]  # Invert Y-axis
        return (x, y)
    
    # ===========================================
    # Main Processing
    # ===========================================
    
    def process(self, input_path: str, output_path: Optional[str] = None) -> ProcessingResult:
        """
        Main processing
        
        Args:
            input_path: Input image path
            output_path: Output DXF path (auto-generated if omitted)
            
        Returns:
            Processing result
        """
        import time
        start_time = time.time()
        
        # Auto-generate output path
        if output_path is None:
            input_file = Path(input_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/{input_file.stem}_converted_{timestamp}.dxf"
            
            # Create output directory
            Path("output").mkdir(exist_ok=True)
        
        try:
            logger.info("=" * 50)
            logger.info("SmartCAD Processing Started (GPT-5)")
            logger.info("=" * 50)
            
            # 1. Image preprocessing
            processed = self.preprocess_image(input_path)

            # Debug: save masks if enabled
            if os.getenv("SKETCH2CAD_DEBUG"):
                try:
                    debug_dir = Path("output") / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(debug_dir / f"red_mask_{ts}.png"), processed['red_mask'])
                    cv2.imwrite(str(debug_dir / f"blue_mask_{ts}.png"), processed['blue_mask'])
                    cv2.imwrite(str(debug_dir / f"black_mask_{ts}.png"), processed['black_mask'])
                    logger.info(f"Saved debug masks to {debug_dir}/ (timestamp {ts})")
                except Exception as e:
                    logger.warning(f"Failed to save debug masks: {e}")
            
            # 2. Element detection
            elements = self.detect_elements(processed)
            
            # 3. AI recognition
            warnings = []
            if self.use_ai:
                try:
                    import asyncio
                    ai_elements = asyncio.run(self.recognize_with_ai(processed))
                    elements.extend(ai_elements)
                    logger.info(f"AI recognition enabled: provider={self.ai_provider}, recognized={len(ai_elements)} elements")
                except AIApiError as e:
                    warning_msg = f"AI recognition failed: {e}. Output may be incomplete."
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)
                except Exception as e:
                    logger.error(f"An unexpected error occurred during AI recognition: {e}")
                    warnings.append(f"An unexpected error occurred during AI recognition: {e}")
            else:
                logger.info("AI recognition disabled by user or config.")

            # 4. DXF Conversion
            success = self.convert_to_dxf(elements, output_path, processed['height'])

            processing_time = time.time() - start_time
            
            if success:
                logger.info("=" * 50)
                logger.info("✅ Conversion successful!")
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                logger.info("=" * 50)
                
                return ProcessingResult(
                    success=True,
                    output_path=output_path,
                    element_count=len(elements),
                    processing_time=processing_time,
                    warnings=warnings
                )
            else:
                return ProcessingResult(
                    success=False,
                    error_message="DXF conversion failed",
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )


# ===========================================
# Utility Functions
# ===========================================

def setup_directories():
    """Create necessary directories"""
    dirs = ['input', 'output']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        if not any(Path(dir_name).iterdir()):
            logger.info(f"✅ Created {dir_name}/ directory")


def validate_input(input_path: str) -> bool:
    """Validate input file"""
    if not os.path.exists(input_path):
        logger.error(f"File not found: {input_path}")
        return False
    
    # Check extension
    valid_extensions = ['.png', '.jpg', '.jpeg']
    ext = Path(input_path).suffix.lower()
    if ext not in valid_extensions:
        logger.error(f"Unsupported format: {ext}")
        logger.error(f"Supported formats: {', '.join(valid_extensions)}")
        return False
    
    # Check file size (max 20MB)
    file_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    if file_size > 20:
        logger.error(f"File too large: {file_size:.1f}MB (max: 20MB)")
        return False
    
    return True


# ===========================================
# Main Entry Point
# ===========================================

def main():
    """Command line execution"""
    import argparse
    parser = argparse.ArgumentParser(description="Convert hand-drawn CAD markups to DXF files.")
    parser.add_argument("input_image", help="Path to the input image file (PNG, JPG).")
    parser.add_argument("output_dxf", nargs='?', default=None, help="Optional path for the output DXF file.")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI recognition and run only CV-based detection.")

    args = parser.parse_args()

    # Setup directories
    setup_directories()

    # Validate input
    if not validate_input(args.input_image):
        sys.exit(1)

    # Determine if AI should be used
    use_ai_flag = not args.no_ai

    # Execute conversion
    converter = SketchToCAD(use_ai=use_ai_flag, ai_provider='gpt5')
    result = converter.process(args.input_image, args.output_dxf)
    
    if result.success:
        if result.warnings:
            print("\n⚠️  Conversion complete with warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        else:
            print("\n✅ Conversion complete!")
        
        print(f"📁 Output file: {result.output_path}")
        print(f"📊 Elements detected: {result.element_count}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
    else:
        print(f"\n❌ Conversion failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
