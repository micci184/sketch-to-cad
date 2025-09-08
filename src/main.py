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
from dataclasses import dataclass
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


@dataclass
class ProcessingResult:
    """Data class representing processing result"""
    success: bool
    output_path: Optional[str] = None
    element_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None


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
    
    def detect_elements(self, processed_data: Dict) -> List[CADElement]:
        """
        Detect drawing elements
        
        Args:
            processed_data: Preprocessed data
            
        Returns:
            List of detected elements
        """
        logger.info("Detecting elements...")
        elements = []
        
        # Existing lines (black)
        black_lines = self._detect_lines(
            processed_data['black_mask'],
            color='black',
            layer='0_EXISTING'
        )
        elements.extend(black_lines)
        logger.info(f"  Existing lines: {len(black_lines)}")
        
        # Added lines (red)
        red_lines = self._detect_lines(
            processed_data['red_mask'],
            color='red',
            layer='1_ADDITION'
        )
        elements.extend(red_lines)
        logger.info(f"  Added lines: {len(red_lines)}")
        
        # Deletion marks
        deletions = self._detect_deletion_marks(processed_data['red_mask'])
        elements.extend(deletions)
        logger.info(f"  Deletion marks: {len(deletions)}")
        
        # Supplementary lines (blue)
        if np.any(processed_data['blue_mask']):
            blue_lines = self._detect_lines(
                processed_data['blue_mask'],
                color='blue',
                layer='3_ANNOTATION'
            )
            elements.extend(blue_lines)
            logger.info(f"  Supplementary lines: {len(blue_lines)}")
        
        return elements
    
    def _detect_lines(self, mask: np.ndarray, color: str, layer: str) -> List[CADElement]:
        """Detect lines using Hough transform"""
        # Adjust parameters by color
        if color == 'black':
            min_length = 50  # Longer lines for CAD
            max_gap = 3      # Smaller gap for precision
            threshold = 50
        else:  # Handwritten lines
            min_length = 25  # Longer lines for handwriting
            max_gap = 15     # Wider gap for broken lines
            threshold = 25   # Lower threshold for faint lines
        
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi/180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Exclude lines that are too short
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length < 15:
                    continue
                
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
        
        try:
            if self.ai_provider == 'gpt5':
                return await self._call_gpt5_api(image_data)
            elif self.ai_provider == 'claude':
                return await self._call_claude_api(image_data)
            elif self.ai_provider == 'gemini':
                return await self._call_gemini_api(image_data)
        except Exception as e:
            import traceback
            logger.error(f"AI recognition error: {e}\n{traceback.format_exc()}")
            return []
        
        return []
    
    async def _call_gpt5_api(self, image_data: Dict) -> List[CADElement]:
        """GPT-5 API call (August 2025 latest)"""
        try:
            from openai import OpenAI
            import base64
            import os

            # Temporarily unset proxy env variables to fix httpx/openai init issue
            proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            original_proxies = {key: os.environ.pop(key, None) for key in proxy_keys}

            try:
                client = OpenAI(api_key=self.api_key)
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
                                "text": """Analyze this architectural drawing and identify:
                                1. Red pen additions/modifications
                                2. Blue pen annotations
                                3. Deletion marks (X marks)
                                4. Japanese text annotations
                                
                                Output in JSON format with coordinates and types."""
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
            
            # Parse JSON and create CADElements
            import json
            import re
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                elements = []
                
                # Process text elements
                for text_item in data.get('texts', []):
                    element = CADElement(
                        element_type='text',
                        coordinates=[(text_item['x'], text_item['y'])],
                        content=text_item.get('content', ''),
                        layer='3_ANNOTATION',
                        confidence=0.9
                    )
                    elements.append(element)
                
                logger.info(f"  GPT-5 recognized {len(elements)} elements")
                return elements
                
        except Exception as e:
            import traceback
            logger.error(f"GPT API error: {e}\n{traceback.format_exc()}")
            
            return []
    
    async def _call_claude_api(self, image_data: Dict) -> List[CADElement]:
        """Claude API call (optional)"""
        # For future extension
        return []
    
    async def _call_gemini_api(self, image_data: Dict) -> List[CADElement]:
        """Gemini API call (optional)"""
        # For future extension
        return []
    
    # ===========================================
    # DXF Conversion
    # ===========================================
    
    def convert_to_dxf(self, elements: List[CADElement], output_path: str) -> bool:
        """
        Convert to DXF file
        
        Args:
            elements: Detected elements
            output_path: Output file path
            
        Returns:
            Success/Failure
        """
        logger.info(f"Starting DXF conversion...")
        
        try:
            # Create DXF document
            doc = ezdxf.new('R2018')
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
                if self._add_element_to_dxf(msp, element):
                    added_count += 1
            
            # Save file
            doc.saveas(output_path)
            logger.info(f"  DXF saved: {output_path}")
            logger.info(f"  Elements added: {added_count}/{len(elements)}")
            
            return True
            
        except Exception as e:
            logger.error(f"DXF conversion error: {e}")
            return False
    
    def _add_element_to_dxf(self, msp, element: CADElement) -> bool:
        """Add element to DXF model space"""
        try:
            if element.element_type == 'line':
                if len(element.coordinates) >= 2:
                    start = self._pixel_to_mm(element.coordinates[0])
                    end = self._pixel_to_mm(element.coordinates[1])
                    
                    msp.add_line(
                        start=start,
                        end=end,
                        dxfattribs={'layer': element.layer}
                    )
                    return True
            
            elif element.element_type == 'deletion':
                if element.coordinates:
                    center = self._pixel_to_mm(element.coordinates[0])
                    size = 10.0  # mm
                    
                    # Draw X mark
                    msp.add_line(
                        start=(center[0]-size/2, center[1]-size/2),
                        end=(center[0]+size/2, center[1]+size/2),
                        dxfattribs={'layer': element.layer, 'color': 1}
                    )
                    msp.add_line(
                        start=(center[0]-size/2, center[1]+size/2),
                        end=(center[0]+size/2, center[1]-size/2),
                        dxfattribs={'layer': element.layer, 'color': 1}
                    )
                    return True
            
            elif element.element_type == 'text' and element.content:
                if element.coordinates:
                    position = self._pixel_to_mm(element.coordinates[0])
                    
                    msp.add_text(
                        element.content,
                        height=2.5,
                        dxfattribs={
                            'layer': element.layer,
                            'insert': position
                        }
                    )
                    return True
                    
        except Exception as e:
            logger.warning(f"Element add error: {e}")
            
        return False
    
    def _pixel_to_mm(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        """Convert pixel coordinates to mm"""
        # A3: 297×420mm, assuming 300dpi
        scale = 0.0847  # mm/pixel (25.4mm/inch ÷ 300dpi)
        x = coord[0] * scale
        y = 297 - coord[1] * scale  # Y-axis inversion (CAD coordinate system)
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
            
            # 3. AI recognition (if async processing needed)
            ai_elements: List[CADElement] = []
            if self.use_ai:
                import asyncio
                ai_elements = asyncio.run(self.recognize_with_ai(processed))
                elements.extend(ai_elements)
                logger.info(f"AI recognition enabled: provider={self.ai_provider}, recognized={len(ai_elements)} elements")
            else:
                logger.info("AI recognition disabled (no API key or provider not configured)")
            
            # 4. DXF conversion
            success = self.convert_to_dxf(elements, output_path)
            
            processing_time = time.time() - start_time
            
            if success:
                logger.info("=" * 50)
                logger.info(f"✅ Conversion successful!")
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                logger.info("=" * 50)
                
                return ProcessingResult(
                    success=True,
                    output_path=output_path,
                    element_count=len(elements),
                    processing_time=processing_time
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
    # Setup directories
    setup_directories()
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python src/main.py <input_image> [output_dxf]")
        print("\nExamples:")
        print("  python src/main.py input/drawing.png")
        print("  python src/main.py input/scan.jpg output/result.dxf")
        print("\nSupported formats: PNG, JPG")
        print("Recommended: Scan with iPhone Notes app → Save as PNG")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate input
    if not validate_input(input_path):
        sys.exit(1)
    
    # Execute conversion (using GPT-5)
    converter = SketchToCAD(use_ai=True, ai_provider='gpt5')
    result = converter.process(input_path, output_path)
    
    if result.success:
        print(f"\n✅ Conversion complete!")
        print(f"📁 Output file: {result.output_path}")
        print(f"📊 Elements detected: {result.element_count}")
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
    else:
        print(f"\n❌ Conversion failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
