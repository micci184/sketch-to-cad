"""
Sketch-to-CAD Setup Script (GPT-5 Version)
Check dependencies and environment configuration
"""

import sys


def check_dependencies():
    """Check if required packages are installed"""
    print("=" * 50)
    print("Sketch-to-CAD Environment Check (GPT-5)")
    print("=" * 50)
    print("\n📦 Checking required packages...")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'ezdxf': 'ezdxf'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print("\n❌ Missing packages detected")
        print("\nPlease run the following command:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages installed")
    return True


def check_ai_packages():
    """Check AI API packages"""
    print("\n🤖 Checking AI API packages...")
    
    # GPT-5 (Primary)
    try:
        import openai
        print(f"  ✅ GPT-5 API (openai)")
        gpt5_ok = True
    except ImportError:
        print(f"  ❌ GPT-5 API (openai) - Not installed")
        print("     pip install openai")
        gpt5_ok = False
    
    # Others (Optional)
    optional = {
        'anthropic': 'Claude API',
        'google.generativeai': 'Gemini API'
    }
    
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  - {name} (optional)")
        except ImportError:
            pass  # Optional, so ignore
    
    return gpt5_ok


def check_environment():
    """Check environment variables"""
    import os
    
    print("\n🔑 Checking environment variables...")
    
    # Check GPT-5 API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"  ✅ OPENAI_API_KEY (configured)")
        print(f"     Same account as ChatGPT Plus")
        return True
    else:
        print(f"  ❌ OPENAI_API_KEY (not configured)")
        
    # Other API keys (optional)
    other_keys = {
        'CLAUDE_API_KEY': 'Claude API',
        'GOOGLE_AI_KEY': 'Gemini API'
    }
    
    for var, name in other_keys.items():
        if os.getenv(var):
            print(f"  - {var} ({name} - optional)")
    
    print("\n⚠️  OpenAI API key not configured")
    print("\nHow to set it up:")
    print("  export OPENAI_API_KEY='sk-xxxxx'")
    print("\n※ You can use the same account as ChatGPT Plus")
    return False


def print_usage():
    """Display usage instructions"""
    print("\n" + "=" * 50)
    print("✅ Setup Complete!")
    print("=" * 50)
    
    print("\n📖 Usage:")
    print("  python src/main.py input/drawing.png")
    print("  python src/main.py input/scan.jpg output/result.dxf")
    
    print("\n💡 Recommended Workflow:")
    print("  1. Scan drawing with iPhone Notes app")
    print("  2. Save as PNG → Place in input/ folder")
    print("  3. Run the command above")
    
    print("\n📋 Supported formats: PNG, JPG (300+ DPI recommended)")
    
    print("\n💰 Cost Estimation (GPT-5):")
    print("  - Per image: ~$0.005")
    print("  - $5 credit: ~1000 images")
    print("  - Monthly (100 images): ~$0.50")


def main():
    """Main process"""
    # Check required packages
    deps_ok = check_dependencies()
    
    # Check AI packages
    ai_ok = check_ai_packages()
    
    # Check environment variables
    env_ok = check_environment()
    
    # Display results
    if deps_ok and ai_ok and env_ok:
        print_usage()
        print("\n🚀 Ready for high-accuracy, cost-effective conversion with GPT-5!")
    else:
        print("\n⚠️  Setup incomplete")
        print("  Please follow the instructions above to complete setup")
        sys.exit(1)


if __name__ == "__main__":
    main()
