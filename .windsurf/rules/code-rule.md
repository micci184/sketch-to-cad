---
trigger: always_on
---

# .windsurf/rules.yaml
# WindsurfのAI Agentに対する開発ルール・指示

project_name: Sketch-to-CAD
description: 手書き修正図面を高精度でDXF形式に変換するAI駆動ツール
version: 1.0.0

# AI Agentへの基本指示
general_instructions:
  - このプロジェクトは建築図面の手書き修正をDXFに変換するAIツールです
  - Vision AI（Claude Opus 4.1、GPT-5、Gemini 2.0 Flash）を必須として使用します
  - コードは明確で保守しやすく、エラーハンドリングを適切に実装してください
  - 日本語の手書き文字認識が重要な要件です
  - **src/sample_implementation.py を実装の参考にしてください**

# コード参照
reference_code:
  sample_file: src/sample_implementation.py
  description: |
    このファイルには以下の実装パターンが含まれています：
    - クラス構造（SketchToCAD）
    - エラーハンドリングパターン
    - ログ出力の標準形式
    - 型ヒントの使用方法
    - docstringの記述方法
    - APIキーの管理方法
  
  important_patterns:
    - データクラスの使用（DetectedElement, ProcessingResult）
    - 環境変数からのAPIキー読み込み
    - 段階的な処理フロー（前処理→検出→AI認識→DXF変換）
    - 適切なログレベルの使用

# コーディング規約
coding_standards:
  language: Python 3.8+
  style_guide: PEP 8
  naming_conventions:
    - 関数名: snake_case（例: process_image, convert_to_dxf）
    - クラス名: PascalCase（例: SketchToCAD, DXFConverter）
    - 定数: UPPER_SNAKE_CASE（例: MAX_IMAGE_SIZE, DEFAULT_DPI）
  documentation:
    - すべての関数にdocstringを記述（Google Style）
    - 複雑なロジックには日本語コメントを追加
    - 型ヒントを必ず使用
  code_quality:
    - 関数は単一責任の原則に従う
    - 1関数は50行以内
    - ネストは3段階まで
    - 早期returnを活用

# プロジェクト構造
project_structure:
  directories:
    src/: メインソースコード
    tests/: テストコード
    data/: テスト用画像
    output/: 生成されたDXFファイル
    config/: 設定ファイル
  main_files:
    - src/smartcad.py: メインエントリーポイント
    - src/image_processor.py: 画像前処理
    - src/ai_recognizer.py: AI認識エンジン
    - src/dxf_converter.py: DXF変換
    - src/utils.py: ユーティリティ関数

# AI Vision API統合
ai_integration:
  required_apis:
    - Claude Opus 4.1（構造理解・総合認識）
    - GPT-5（高速処理・コスト効率）
    - Gemini 2.0 Flash（日本語OCR特化）
  
  api_usage_rules:
    - APIキーは必ず環境変数から読み込む
    - エラー時は3回までリトライ
    - レート制限を考慮（sleep実装）
    - API応答をキャッシュして無駄な呼び出しを避ける
    - 費用対効果を考慮してモデルを選択
  
  model_selection:
    simple_lines: GPT-5（安価で高速）
    complex_structure: Claude Opus 4.1（高精度）
    japanese_text: Gemini 2.0 Flash（日本語特化）

# 画像処理ルール
image_processing:
  preprocessing:
    - 入力画像は300DPI以上を推奨
    - 最大サイズ: 20MB
    - 対応形式: JPG, PNG, PDF
  
  color_detection:
    red_pen: [0, 50, 50] to [10, 255, 255]  # HSV範囲
    blue_pen: [100, 50, 50] to [130, 255, 255]
    black_ink: threshold < 100
  
  enhancement:
    - コントラスト強調はCLAHEを使用
    - ノイズ除去は最小限に（手書き文字を保護）
    - 歪み補正は必要時のみ適用

# DXF出力ルール
dxf_output:
  format: AutoCAD 2018 (R2018)
  encoding: UTF-8
  units: millimeters
  coordinate_system: 左上原点、Y軸下向き
  
  layer_structure:
    0_EXISTING:
      color: 7
      description: 既存CAD要素
    1_ADDITION:
      color: 1
      description: 手書き追加要素（赤ペン）
    2_DELETION:
      color: 6
      description: 削除指示（×印）
    3_ANNOTATION:
      color: 2
      description: 手書き注記・テキスト
    9_REVIEW:
      color: 4
      description: 確認が必要な要素

# エラーハンドリング
error_handling:
  api_errors:
    - 一時的なエラーは3回リトライ
    - APIキー無効時は明確なエラーメッセージ
    - レート制限時は適切な待機時間
  
  image_errors:
    - 読み込み失敗時は詳細なエラー情報
    - 解像度不足時は警告表示
    - 形式非対応時は変換を提案
  
  recognition_errors:
    - 認識信頼度が低い要素は9_REVIEWレイヤーへ
    - 部分的失敗でも処理継続
    - ログに詳細を記録

# パフォーマンス目標
performance_targets:
  processing_time: < 20秒/画像
  accuracy_target: > 85%
  api_cost: < $0.05/画像
  memory_usage: < 2GB

# テスト要件
testing_requirements:
  unit_tests:
    - すべての主要関数にテスト作成
    - モックを使用してAPI呼び出しをテスト
    - エッジケースを必ずカバー
  
  integration_tests:
    - 実際の手書き図面でE2Eテスト
    - 各AIモデルの統合テスト
    - DXF出力の妥当性検証

# デバッグ・ログ
debugging:
  logging_level: INFO（本番）、DEBUG（開発）
  log_format: "[%(asctime)s] %(levelname)s: %(message)s"
  debug_outputs:
    - 中間処理画像を保存
    - API リクエスト/レスポンスを記録
    - 処理時間の詳細計測

# セキュリティ
security:
  - APIキーをコードにハードコードしない
  - アップロードファイルのバリデーション実施
  - 一時ファイルは処理後に削除
  - ユーザー入力のサニタイズ

# 最適化のヒント
optimization_tips:
  - 画像が大きい場合は適切にリサイズ
  - バッチ処理で複数画像を効率的に処理
  - プロンプトキャッシングで費用削減
  - 並列処理で処理時間短縮

# 実装の優先順位
implementation_priority:
  phase_1:
    - 基本的な画像読み込みとDXF出力
    - GPT-5を使った線分認識
    - 色分離（赤・黒）
  
  phase_2:
    - Claude/Geminiの統合
    - 日本語テキスト認識
    - 削除マーク検出
  
  phase_3:
    - 精度向上の最適化
    - バッチ処理機能
    - Web UI（オプション）

# 禁止事項
do_not:
  - グローバル変数を使用しない
  - try-except で例外を握りつぶさない
  - 同期的に複数のAPI呼び出しを行わない（並列化する）
  - テストなしでコードをコミットしない

# よく使うコードパターン
common_patterns:
  api_call_with_retry: |
    import time
    from typing import Optional
    
    def call_api_with_retry(func, max_retries=3):
        for i in range(max_retries):
            try:
                return func()
            except RateLimitError:
                time.sleep(2 ** i)
            except Exception as e:
                if i == max_retries - 1:
                    raise
        return None
  
  image_preprocessing: |
    import cv2
    import numpy as np
    
    def preprocess_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        # コントラスト強調
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        return img

# AI Agentへの追加指示
additional_instructions:
  - ユーザーが質問した際は、このプロジェクトの文脈を理解して回答
  - コード生成時は必ずエラーハンドリングを含める
  - 日本語ドキュメントとコメントを重視
  - 実装は段階的に、テスト可能な単位で進める
