# Sketch-to-CAD

AI-powered tool to convert hand-drawn CAD markups to DXF files

## Overview

Automatically detects handwritten modifications (red/blue pen) on architectural drawings and converts them to DXF format CAD files. Achieves 85-95% accuracy using GPT-5 Vision API (August 2025 latest model).

## Requirements

- Python 3.8+
- OpenCV 4.8+
- Image format: PNG/JPG (300+ DPI recommended)

## Installation

```bash
# Clone repository
git clone https://github.com/micci184/sketch-to-cad.git
cd sketch-to-cad

# Install dependencies
pip install -r requirements.txt

# Check environment
python setup.py
```

## Setup

Set OpenAI API key as environment variable:

```bash
export OPENAI_API_KEY='sk-xxxxx'
```

Note: This uses the same account as your ChatGPT Plus subscription.

## Usage

### Basic Usage

```bash
# Convert with auto-generated output
python src/main.py input/drawing.png

# Specify output path
python src/main.py input/scan.jpg output/result.dxf
```

## Directory Structure

```
sketch-to-cad/
├── src/
│   ├── main.py                    # Main program (CLI entrypoint)
│   └── sample_implementation.py   # Reference implementation
├── input/                         # Place input images here
├── output/                        # DXF files output here
├── .windsurf/
│   └── rules/
│       └── code-rule.md           # Windsurf AI configuration
├── setup.py                       # Environment check
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Features

### Auto-Detection

- **Red pen**: Additions/modifications
- **Blue pen**: Supplementary notes
- **Black lines**: Existing CAD elements
- **× marks**: Deletion marks

### DXF Layer Structure

- `0_EXISTING`: Original CAD elements
- `1_ADDITION`: Red pen additions
- `2_DELETION`: Deletion marks
- `3_ANNOTATION`: Notes and annotations

## Specifications

### Input

- Format: PNG, JPG
- Resolution: 300+ DPI recommended
- Size: Max 20MB

### Output

- Format: DXF (AutoCAD 2018 compatible)
- Units: Millimeters
- Coordinate system: Top-left origin

## AI API

Primary:

- **GPT-5** (August 2025) - Latest OpenAI model with excellent vision capabilities and cost efficiency
  - Best for: Overall accuracy and speed
  - Cost: ~$0.005 per image
  - Same account as ChatGPT Plus

Future extensions (optional):

- Claude Opus 4.1 - High-precision structure understanding
- Gemini 2.0 Flash - Japanese OCR specialized

## Development

### Testing

```bash
# Run basic test
python src/main.py input/test.png

# Check output
ls -la output/
```

## License

MIT License
