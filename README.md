# 🎯 ParsaCV - Advanced Multilingual CV Analyzer

<div align="center">

![ParsaCV Logo](https://img.shields.io/badge/ParsaCV-Advanced%20CV%20Analyzer-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Intelligent CV screening with advanced NLP and semantic matching**
**فحص ذكي للسير الذاتية باستخدام معالجة اللغة الطبيعية المتقدمة والمطابقة الدلالية**

</div>

## 🚀 Features | المميزات

### 🔍 **Advanced Information Extraction**
- **🤖 NLP Pattern Matching**: Uses spaCy with custom patterns for accurate extraction
- **📅 Smart Date Parsing**: Recognizes multiple date formats and calculates experience duration
- **🏢 Company & Location Detection**: Extracts company names and geographical entities
- **🎓 Education Analysis**: Comprehensive degree and institution extraction

### 🧠 **Semantic Matching Technology**
- **🔗 Semantic Similarity**: Uses sentence-transformers for meaning-based matching
- **📊 Hybrid Scoring**: Combines semantic and keyword-based approaches
- **🎯 Context Understanding**: Matches "Software Engineer" with "Software Developer"
- **📈 Confidence Scoring**: Provides reliability metrics for each analysis

### 🌍 **Bilingual Support**
- **🇺🇸 English & 🇸🇦 Arabic**: Full RTL and LTR text processing
- **🔄 Auto Language Detection**: Automatically detects CV language
- **🌐 Multilingual Interface**: Switch between English and Arabic UI
- **📝 Cross-language Analysis**: Mix CVs in different languages

### 📄 **Multi-format Processing**
- **📋 PDF**: Direct text extraction with layout preservation
- **📝 DOCX**: Complete document parsing including tables
- **🖼️ Images (PNG/JPG)**: Advanced OCR with image preprocessing
- **🔧 Smart Fallback**: Multiple extraction methods for reliability

### 📊 **Professional Reporting**
- **📈 Excel Reports**: Formatted with conditional coloring and statistics
- **📋 Summary Dashboard**: Key metrics and candidate rankings
- **🏆 Top Candidates**: Automatic sorting by match scores
- **💾 Export Options**: Download comprehensive analysis reports

## 🛠️ Installation | التثبيت

### Prerequisites | المتطلبات الأساسية
- Python 3.9 or higher
- pip package manager
- Git (optional, for cloning)

### Quick Setup | الإعداد السريع

```bash
# Clone the repository | استنسخ المستودع
git clone https://github.com/yourusername/parsacv.git
cd parsacv

# Install dependencies | ثبت الاعتماديات
pip install -r requirements.txt

# Download spaCy language models | حمّل نماذج spaCy اللغوية
python -m spacy download en_core_web_sm
python -m spacy download ar_core_news_sm

# Run the application | شغّل التطبيق
streamlit run parsacv.py
```

### Manual Installation | التثبيت اليدوي

```bash
# Core dependencies
pip install streamlit pandas numpy openpyxl
pip install PyMuPDF python-docx Pillow pytesseract
pip install spacy langdetect sentence-transformers
pip install scikit-learn opencv-python python-dateutil

# Download language models
python -m spacy download en_core_web_sm
python -m spacy download ar_core_news_sm
```

### Additional Setup | إعدادات إضافية

#### Windows Users:
- **Tesseract OCR**: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Add Tesseract to your PATH environment variable

#### macOS Users:
```bash
brew install tesseract
```

#### Linux Users:
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-ara  # For Arabic OCR
```

## 🚀 Usage | الاستخدام

### Basic Usage | الاستخدام الأساسي

1. **Start the application | ابدأ التطبيق**
   ```bash
   streamlit run parsacv.py
   ```

2. **Upload CV files | ارفع ملفات السيرة الذاتية**
   - Supported formats: PDF, DOCX, PNG, JPG, JPEG
   - Multiple files supported

3. **Enter Job Description | أدخل وصف الوظيفة**
   - Paste the complete job requirements
   - Include required skills and qualifications

4. **Analyze & Download | حلل وحمّل**
   - Click "Analyze CVs" button
   - Download the comprehensive Excel report

### Advanced Features | المميزات المتقدمة

#### Language Selection | اختيار اللغة
- Switch between English and Arabic interface
- Automatic CV language detection
- Mixed language support

#### Semantic Matching | المطابقة الدلالية
- Advanced AI understanding of job requirements
- Context-aware skill matching
- Confidence scoring for reliability

#### Detailed Analytics | التحليلات المفصلة
- Work experience timeline analysis
- Education background verification
- Skills gap identification

## 📊 Output Format | صيغة النتائج

### Excel Report Columns | أعمدة تقرير Excel

| English | العربية | Description |
|---------|---------|-------------|
| Full Name | الاسم الكامل | Extracted candidate name |
| Email Address | البريد الإلكتروني | Contact email |
| Phone Number | رقم الهاتف | Phone contact |
| Total Experience | إجمالي الخبرة | Years and months of experience |
| Match Score (%) | نسبة التطابق (%) | Semantic matching percentage |
| Matched Skills | المهارات المطابقة | Skills found in both CV and JD |
| Missing Skills | المهارات المفقودة | Required skills not found |
| Work Experience | الخبرة العملية | Formatted experience details |
| Education | التعليم | Academic background |

### Summary Statistics | الإحصائيات الملخصة
- Total CVs analyzed
- Average match scores
- Top performing candidates
- Language distribution

## 🔧 Configuration | التكوين

### Performance Tuning | تحسين الأداء

```python
# Memory optimization for large batches
import psutil
import gc

# Monitor memory usage
memory_percent = psutil.virtual_memory().percent
if memory_percent > 80:
    gc.collect()
```

### Custom Patterns | الأنماط المخصصة

You can customize extraction patterns by modifying the `setup_patterns()` method:

```python
# Add custom job title patterns
custom_job_patterns = [
    [{"LOWER": "senior"}, {"LOWER": "data"}, {"LOWER": "scientist"}],
    [{"LOWER": "machine"}, {"LOWER": "learning"}, {"LOWER": "engineer"}]
]
```

## 🧪 Testing | الاختبار

### Run Tests | تشغيل الاختبارات
```bash
pytest tests/
```

### Sample Data | بيانات عينة
Test the application with sample CVs included in the `samples/` directory.

## 🤝 Contributing | المساهمة

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup | إعداد التطوير
```bash
# Install development dependencies
pip install pytest black flake8

# Format code
black parsacv.py

# Run linting
flake8 parsacv.py
```

## 📝 License | الترخيص

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments | الشكر والتقدير

- **spaCy** for advanced NLP capabilities
- **Sentence Transformers** for semantic similarity
- **Streamlit** for the excellent web framework
- **OpenCV** for image preprocessing
- The open-source community for their invaluable contributions

## 📞 Support | الدعم

For support, email support@parsacv.com or create an issue on GitHub.

## 🔮 Roadmap | خارطة الطريق

- [ ] **GPT Integration**: Advanced AI analysis
- [ ] **Mobile App**: iOS and Android versions
- [ ] **Cloud Deployment**: AWS/Azure hosting
- [ ] **ATS Integration**: Connect with existing systems
- [ ] **Email Notifications**: Automated result delivery
- [ ] **Multi-language Support**: Add more languages
- [ ] **API Endpoints**: RESTful API for integration
- [ ] **Batch Processing**: Handle thousands of CVs

---

<div align="center">

**Made with ❤️ for HR professionals worldwide**
**صُنع بـ ❤️ لمحترفي الموارد البشرية حول العالم**

[⭐ Star this repo](https://github.com/yourusername/parsacv) • [🐛 Report Bug](https://github.com/yourusername/parsacv/issues) • [💡 Request Feature](https://github.com/yourusername/parsacv/issues)

</div>