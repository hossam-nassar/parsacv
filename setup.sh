#!/bin/bash

# ParsaCV - Quick Setup Script
# This script automates the installation process

echo "🎯 ParsaCV - Advanced Multilingual CV Analyzer"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check Python version
print_step "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python $python_version is compatible ✅"
else
    print_error "Python 3.9+ is required. Current version: $python_version"
    exit 1
fi

# Check if pip is installed
print_step "Checking pip installation..."
if command -v pip3 &> /dev/null; then
    print_status "pip3 is available ✅"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    print_status "pip is available ✅"
    PIP_CMD="pip"
else
    print_error "pip is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    print_step "Creating virtual environment..."
    python3 -m venv parsacv_env
    source parsacv_env/bin/activate
    print_status "Virtual environment created and activated ✅"
fi

# Upgrade pip
print_step "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install requirements
print_step "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
    print_status "Dependencies installed successfully ✅"
else
    print_warning "requirements.txt not found. Installing manually..."
    
    # Core dependencies
    $PIP_CMD install streamlit>=1.28.0
    $PIP_CMD install pandas>=2.0.0 numpy>=1.24.0 openpyxl>=3.1.0
    
    # Document processing
    $PIP_CMD install PyMuPDF>=1.23.0 python-docx>=0.8.11 Pillow>=10.0.0
    
    # OCR and NLP
    $PIP_CMD install pytesseract>=0.3.10 spacy>=3.7.0 langdetect>=1.0.9
    
    # ML and semantic similarity
    $PIP_CMD install sentence-transformers>=2.2.2 scikit-learn>=1.3.0
    
    # Computer vision
    $PIP_CMD install opencv-python>=4.8.0
    
    # Utilities
    $PIP_CMD install python-dateutil>=2.8.2
    
    print_status "Manual installation completed ✅"
fi

# Download spaCy models
print_step "Downloading spaCy language models..."
print_status "Downloading English model..."
python3 -m spacy download en_core_web_sm

print_status "Downloading Arabic model..."
python3 -m spacy download ar_core_news_sm

# Check Tesseract installation
print_step "Checking Tesseract OCR installation..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version | head -n1)
    print_status "Tesseract found: $tesseract_version ✅"
else
    print_warning "Tesseract OCR not found. Image processing will be limited."
    echo "To install Tesseract:"
    echo "  - Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "  - macOS: brew install tesseract"
    echo "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Create sample directory structure
print_step "Setting up directory structure..."
mkdir -p samples
mkdir -p outputs
mkdir -p temp_cvs

# Create a simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test script to verify ParsaCV installation
"""

def test_imports():
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import spacy
        print("✅ spaCy imported successfully")
    except ImportError as e:
        print(f"❌ spaCy import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    return True

def test_spacy_models():
    try:
        import spacy
        nlp_en = spacy.load("en_core_web_sm")
        print("✅ English spaCy model loaded successfully")
    except OSError as e:
        print(f"❌ English spaCy model failed: {e}")
        return False
    
    try:
        nlp_ar = spacy.load("ar_core_news_sm")
        print("✅ Arabic spaCy model loaded successfully")
    except OSError as e:
        print(f"❌ Arabic spaCy model failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing ParsaCV Installation")
    print("=" * 40)
    
    if test_imports():
        print("\n📦 All required packages imported successfully")
    else:
        print("\n❌ Some packages failed to import")
        exit(1)
    
    if test_spacy_models():
        print("\n🧠 All language models loaded successfully")
    else:
        print("\n❌ Some language models failed to load")
        exit(1)
    
    print("\n🎉 Installation test passed! ParsaCV is ready to use.")
    print("\nTo start the application, run:")
    print("  streamlit run parsacv.py")
EOF

# Make test script executable
chmod +x test_installation.py

# Run installation test
print_step "Running installation test..."
python3 test_installation.py

if [ $? -eq 0 ]; then
    print_status "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run the application: streamlit run parsacv.py"
    echo "2. Open your browser to the URL shown (usually http://localhost:8501)"
    echo "3. Upload some CV files and a job description to test"
    echo ""
    echo "For help and documentation, check the README.md file"
else
    print_error "Setup encountered some issues. Please check the error messages above."
fi

# Cleanup
rm -f test_installation.py