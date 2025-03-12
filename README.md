# Text Summarizer

A deep learning-based application for automatic text summarization using Transformer models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Training the Model](#training-the-model)
  - [Generating Summaries](#generating-summaries)
- [Pipeline Components](#pipeline-components)
- [Technical Stack](#technical-stack)
- [License](#license)

## Overview

This application leverages state-of-the-art transformer models to create concise and accurate summaries of text documents. Built with a modular architecture, the system is designed to handle the complete ML lifecycle from data ingestion to model deployment.

## Features

- End-to-end ML pipeline for text summarization
- FastAPI-based RESTful service for real-time summarization
- Modular architecture with separate components for data processing, model training, and evaluation
- Configurable system parameters via YAML configuration files
- Comprehensive logging and error handling

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd text-summarizer
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

4. Install the package in development mode
```bash
pip install -e .
```

## Project Structure

The project follows a structured organization:

```
text-summarizer/
├── .github/workflows/      # CI/CD workflows
├── config/                 # Configuration files
├── research/               # Jupyter notebooks for experimentation
├── src/textSummarizer/     # Main package
│   ├── components/         # Pipeline components
│   ├── config/             # Configuration handling
│   ├── constants/          # Constants and fixed values
│   ├── entity/             # Data models and structures
│   ├── logging/            # Logging functionality
│   ├── pipeline/           # Pipeline stages
│   └── utils/              # Utility functions
├── app.py                  # FastAPI application
├── main.py                 # Main execution script
├── Dockerfile              # Docker configuration
├── params.yaml             # Model parameters
├── requirements.txt        # Dependencies
└── setup.py                # Package installation
```

## Usage

### API Endpoints

The application provides a RESTful API built with FastAPI:

- **GET /** - Redirects to the API documentation
- **GET /train** - Triggers the training pipeline
- **POST /predict** - Generates a summary for input text

Start the API server:
```bash
python app.py
```

Access the API documentation at `http://localhost:8080/docs`

### Training the Model

To train a new model from scratch:

```bash
python main.py
```

This will run the complete pipeline:
1. Data ingestion (downloading and extracting the dataset)
2. Data transformation (tokenization and preprocessing)
3. Model training (fine-tuning the transformer model)
4. Model evaluation (measuring performance with ROUGE metrics)

### Generating Summaries

You can generate summaries through the API or directly:

```python
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

text = "Your long text to summarize..."
pipeline = PredictionPipeline()
summary = pipeline.predict(text)
print(summary)
```

## Pipeline Components

The application is built with a modular pipeline architecture:

1. **Data Ingestion**: Downloads and extracts the dataset
2. **Data Transformation**: Tokenizes and preprocesses the text data
3. **Model Training**: Fine-tunes a pretrained transformer model
4. **Model Evaluation**: Calculates ROUGE metrics to evaluate performance
5. **Prediction Pipeline**: Handles real-time summarization requests

## Technical Stack

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for transformer models
- **FastAPI**: Web framework for API
- **ROUGE**: Evaluation metrics
- **Pandas**: Data manipulation
- **NLTK**: Natural language processing utilities
- **YAML**: Configuration management
- **Uvicorn**: ASGI server

## License
Hussein Baghdadi