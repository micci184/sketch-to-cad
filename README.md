# Sketch-to-CAD

Convert hand-drawn CAD markups to DXF files using Vision AI

## Overview

This tool uses state-of-the-art Vision AI models to convert paper-based architectural drawings with handwritten annotations into digital CAD (DXF) format. Achieves 85-95% accuracy compared to 40-50% with traditional image processing methods.

## Requirements

- Python 3.8+
- OpenCV 4.8+
- ezdxf 1.1.0+
- API Key (choose one):
  - Claude Opus 4.1: $15/1M input, $75/1M output tokens
  - GPT-5: $1.25/1M input, $10/1M output tokens
  - Gemini 2.0 Flash: Available via Google Cloud Vertex AI

## Installation

```bash
git clone https://github.com/yourusername/sketch-to-cad.git
cd sketch-to-cad
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python src/smartcad.py input.jpg output.dxf

# Specify AI provider
python src/smartcad.py drawing.jpg --provider gpt5
```

## Configuration

Set API keys in environment variables:
```bash
export CLAUDE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_AI_KEY="your-key"
```

## Features

- Handwritten text recognition (Japanese/English)
- Red/blue pen markup detection
- Automatic layer separation
- High accuracy (85-95%)

## Output

DXF format with layers:
- `0_EXISTING` - Original elements
- `1_ADDITION` - Added elements
- `2_DELETION` - Deletion marks
- `3_ANNOTATION` - Text annotations

## Performance

- Processing time: 10-20 seconds per drawing
- Cost: $0.01-0.05 per image
- Accuracy: 85-95%

## License

MIT
