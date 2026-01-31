# 🎯 Skin Problem Detection & Advisor App

A YOLO-based Object Detection application to detect various skin conditions in real-time, integrated with an AI Skin Assistant for personalized advice and product recommendations using Streamlit and OpenAI.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Poetry](https://img.shields.io/badge/poetry-dependency%20management-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![YOLO](https://img.shields.io/badge/YOLO-ultralytics-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)

## 🩺 Supported Detections

The model is trained to detect 10 types of skin conditions:
- **Acne**
- **Blackheads**
- **Dark Spots**
- **Dry Skin**
- **Enlarged Pores**
- **Eyebags**
- **Oily Skin**
- **Skin Redness**
- **Whiteheads**
- **Wrinkles**

## 🤖 New Feature: AI Skin Assistant
After the detection process, you can now interact with a built-in **AI Skin Assistant** powered by OpenAI GPT-4o-mini.
- **Contextual Advice**: The assistant receives the detection results to provide relevant skin care tips.
- **HAUM Product Recommendations**: Get specific product suggestions from the [HAUM](https://haum.co.id/) skincare line based on your detected skin concerns.

## 📁 Project Structure

```
SkinAnalyzerRecommendation/
├── models/                          # Folder for .pt model files
│   └── best_skinproblem.pt          # Main skin detection model
├── src/src/                         # Application source code
│   ├── main.py                      # Streamlit application (Entry point)
│   ├── llm_utils.py                 # LLM logic & HAUM product data
│   └── __init__.py
├── training_code/                   # Notebooks for model training
├── tests/                           # Unit tests
├── pyproject.toml                   # Poetry configuration
├── .env                             # Environment variables (API Keys)
└── README.md                        # Project documentation
```

## 🚀 Local Development Setup

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- OpenAI API Key

### Installation Steps

#### 1. Clone & Install

```bash
cd SkinAnalyzerRecommendation
poetry install
```

#### 2. Environment Setup

Create a `.env` file in the root directory and add your OpenAI API Key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### 3. Model Verification

Ensure the model file `.pt` is in the correct location:
- Path: `models/best_skinproblem.pt`

## 🎯 How to Use the App

### Run the Application

Execute the following command in your terminal:

```bash
poetry run streamlit run src/src/main.py
```

### Steps to Use

1.  **Access Browser**: Open `http://localhost:8501`.
2.  **Adjust Confidence**: Use the **Confidence** slider to set the detection sensitivity (recommended starts at 0.25).
3.  **Upload Image**: Upload a face photo or skin area image you want to analyze.
4.  **Analyze**: Click the **🔍 Detect Objects** button.
5.  **View Results**: The app will display the annotated image with detection boxes and a summary of the object counts.
6.  **Chat with AI**: Scroll down to the **Skin Assistant** section to ask follow-up questions or get product recommendations from HAUM.

---
*by techyfeels ❤️ for AI Engineering Project*
