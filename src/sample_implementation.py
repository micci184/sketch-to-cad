"""
sample_implementation.py
AI Agent用の実装サンプル - Sketch-to-CAD

このファイルをプロジェクトに含めることで、
WindsurfのAI Agentが実装パターンを理解しやすくなります。
"""

import os
import cv2
import numpy as np
import ezdxf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================================
# データクラス定義
# ===========================================

@dataclass
class DetectedElement:
    """検出された図面要素"""
    element_type: str  # 'line', 'text', 'symbol', 'deletion'
    coordinates: List[Tuple[float, float]]
    content: Optional[str] = None
    color: str = 'black'
    confidence: float = 1.0
    layer: str = '0_EXISTING'


@dataclass
class ProcessingResult:
    """処理結果"""
    success: bool
    output_path: Optional[str] = None
    element_count: int = 0
    error_message: Optional[str] = None
    processing_time: float = 0.0


# ===========================================
# メインクラス
# ===========================================

class SketchToCAD:
    """手書き図面をDXFに変換するメインクラス"""
    
    def __init__(self, ai_provider: str = "gpt5"):
        """
        初期化
        
        Args:
            ai_provider: 使用するAIプロバイダー ('claude', 'gpt5', 'gemini')
        """
        self.ai_provider = ai_provider
        self.api_key = self._load_api_key(ai_provider)
        self.elements: List[DetectedElement] = []
        
        # DXFレイヤー定義
        self.layers = {
            '0_EXISTING': {'color': 7, 'desc': '既存要素'},
            '1_ADDITION': {'color': 1, 'desc': '追加要素'},
            '2_DELETION': {'color': 6, 'desc': '削除指示'},
            '3_ANNOTATION': {'color': 2, 'desc': '注記'},
            '9_REVIEW': {'color': 4, 'desc': '要確認'}
        }
    
    def _load_api_key(self, provider: str) -> str:
        """APIキーを環境変数から読み込み"""
        key_map = {
            'claude': 'CLAUDE_API_KEY',
            'gpt5': 'OPENAI_API_KEY',
            'gemini': 'GOOGLE_AI_KEY'
        }
        
        env_var = key_map.get(provider)
        api_key = os.getenv(env_var)
        
        if not api_key:
            raise ValueError(f"{env_var}が設定されていません")
        
        return api_key
    
    # ===========================================
    # 画像前処理
    # ===========================================
    
    def preprocess_image(self, image_path: str) -> Dict:
        """
        画像の前処理
        
        Args:
            image_path: 入力画像パス
            
        Returns:
            前処理済み画像データ
        """
        logger.info(f"画像を読み込み中: {image_path}")
        
        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        height, width = img.shape[:2]
        logger.info(f"画像サイズ: {width}x{height}")
        
        # HSV変換で色分離
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 赤ペンの検出
        red_mask = self._detect_red_pen(hsv)
        
        # 青ペンの検出
        blue_mask = self._detect_blue_pen(hsv)
        
        # 黒線の検出
        black_mask = self._detect_black_lines(img)
        
        return {
            'original': img,
            'red_mask': red_mask,
            'blue_mask': blue_mask,
            'black_mask': black_mask,
            'height': height,
            'width': width
        }
    
    def _detect_red_pen(self, hsv: np.ndarray) -> np.ndarray:
        """赤ペンの検出"""
        # 赤色の範囲（HSV）
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        return red_mask
    
    def _detect_blue_pen(self, hsv: np.ndarray) -> np.ndarray:
        """青ペンの検出"""
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        return blue_mask
    
    def _detect_black_lines(self, img: np.ndarray) -> np.ndarray:
        """黒線の検出"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        return black_mask
    
    # ===========================================
    # 要素検出
    # ===========================================
    
    def detect_elements(self, processed_data: Dict) -> List[DetectedElement]:
        """
        図面要素の検出
        
        Args:
            processed_data: 前処理済みデータ
            
        Returns:
            検出された要素リスト
        """
        elements = []
        
        # 既存線分の検出（黒）
        black_lines = self._detect_lines(
            processed_data['black_mask'], 
            color='black',
            layer='0_EXISTING'
        )
        elements.extend(black_lines)
        
        # 追加線分の検出（赤）
        red_lines = self._detect_lines(
            processed_data['red_mask'],
            color='red', 
            layer='1_ADDITION'
        )
        elements.extend(red_lines)
        
        # 削除マークの検出
        deletions = self._detect_deletion_marks(processed_data['red_mask'])
        elements.extend(deletions)
        
        logger.info(f"検出された要素: {len(elements)}個")
        
        return elements
    
    def _detect_lines(self, mask: np.ndarray, color: str, layer: str) -> List[DetectedElement]:
        """ハフ変換による線分検出"""
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                element = DetectedElement(
                    element_type='line',
                    coordinates=[(x1, y1), (x2, y2)],
                    color=color,
                    layer=layer
                )
                detected.append(element)
        
        return detected
    
    def _detect_deletion_marks(self, mask: np.ndarray) -> List[DetectedElement]:
        """削除マーク（×印）の検出"""
        # 簡易的な実装
        # 実際はパターンマッチングやAI認識を使用
        deletions = []
        
        # ここにX印検出ロジックを実装
        
        return deletions
    
    # ===========================================
    # AI認識（スタブ）
    # ===========================================
    
    def recognize_with_ai(self, image_data: Dict) -> List[DetectedElement]:
        """
        AIを使った認識（実装例）
        
        Args:
            image_data: 画像データ
            
        Returns:
            AI認識結果
        """
        logger.info(f"AI認識を実行中 (provider: {self.ai_provider})")
        
        # ここで実際のAI API呼び出しを行う
        # 以下は仮の実装
        
        if self.ai_provider == "gpt5":
            # GPT-5 APIコール
            pass
        elif self.ai_provider == "claude":
            # Claude Opus 4.1 APIコール
            pass
        elif self.ai_provider == "gemini":
            # Gemini 2.0 Flash APIコール
            pass
        
        # 仮の結果
        text_element = DetectedElement(
            element_type='text',
            coordinates=[(100, 100)],
            content='テーブル',
            color='red',
            layer='3_ANNOTATION',
            confidence=0.85
        )
        
        return [text_element]
    
    # ===========================================
    # DXF変換
    # ===========================================
    
    def convert_to_dxf(self, elements: List[DetectedElement], output_path: str) -> bool:
        """
        DXFファイルへの変換
        
        Args:
            elements: 検出された要素
            output_path: 出力ファイルパス
            
        Returns:
            成功/失敗
        """
        logger.info(f"DXF変換を開始: {output_path}")
        
        try:
            # 新規DXFドキュメント作成
            doc = ezdxf.new('R2018')
            msp = doc.modelspace()
            
            # レイヤー作成
            for layer_name, props in self.layers.items():
                doc.layers.add(
                    name=layer_name,
                    color=props['color']
                )
            
            # 要素の追加
            for element in elements:
                self._add_element_to_dxf(msp, element)
            
            # ファイル保存
            doc.saveas(output_path)
            logger.info(f"DXF保存完了: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"DXF変換エラー: {e}")
            return False
    
    def _add_element_to_dxf(self, msp, element: DetectedElement):
        """DXFに要素を追加"""
        if element.element_type == 'line':
            if len(element.coordinates) >= 2:
                start = self._pixel_to_mm(element.coordinates[0])
                end = self._pixel_to_mm(element.coordinates[1])
                
                msp.add_line(
                    start=start,
                    end=end,
                    dxfattribs={'layer': element.layer}
                )
        
        elif element.element_type == 'text':
            if element.coordinates and element.content:
                position = self._pixel_to_mm(element.coordinates[0])
                
                msp.add_text(
                    element.content,
                    height=2.5,
                    dxfattribs={
                        'layer': element.layer,
                        'insert': position
                    }
                )
    
    def _pixel_to_mm(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        """ピクセル座標をmm座標に変換"""
        # A3: 297×420mm, 300dpi想定
        scale = 0.0847  # mm/pixel
        x = coord[0] * scale
        y = 297 - coord[1] * scale  # Y軸反転
        return (x, y)
    
    # ===========================================
    # メイン処理
    # ===========================================
    
    def process(self, input_path: str, output_path: str) -> ProcessingResult:
        """
        メイン処理
        
        Args:
            input_path: 入力画像パス
            output_path: 出力DXFパス
            
        Returns:
            処理結果
        """
        import time
        start_time = time.time()
        
        try:
            # 1. 画像前処理
            processed = self.preprocess_image(input_path)
            
            # 2. 要素検出
            elements = self.detect_elements(processed)
            
            # 3. AI認識（オプション）
            ai_elements = self.recognize_with_ai(processed)
            elements.extend(ai_elements)
            
            # 4. DXF変換
            success = self.convert_to_dxf(elements, output_path)
            
            processing_time = time.time() - start_time
            
            if success:
                return ProcessingResult(
                    success=True,
                    output_path=output_path,
                    element_count=len(elements),
                    processing_time=processing_time
                )
            else:
                return ProcessingResult(
                    success=False,
                    error_message="DXF変換に失敗しました",
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"処理エラー: {e}")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )


# ===========================================
# エントリーポイント
# ===========================================

def main():
    """コマンドライン実行用"""
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python sample_implementation.py <入力画像> <出力DXF>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 変換実行
    converter = SketchToCAD(ai_provider='gpt5')
    result = converter.process(input_path, output_path)
    
    if result.success:
        print(f"✅ 変換成功!")
        print(f"   出力: {result.output_path}")
        print(f"   要素数: {result.element_count}")
        print(f"   処理時間: {result.processing_time:.2f}秒")
    else:
        print(f"❌ 変換失敗: {result.error_message}")


if __name__ == "__main__":
    main()
