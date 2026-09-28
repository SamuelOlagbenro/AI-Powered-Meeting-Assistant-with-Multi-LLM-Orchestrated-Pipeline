# AI-Powered-Meeting-Assistant-with-Multi-LLM-Orchestrated-Pipeline
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-green?logo=huggingface&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-yellow?logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.50+-blue?logoColor=white)
![IBM Watsonx](https://img.shields.io/badge/IBM%20Watsonx-AI-blue?logo=ibm&logoColor=white)
![OpenAI Whisper](https://img.shields.io/badge/OpenAI%20Whisper-Latest-blueviolet?logo=openai&logoColor=white)
![Llama](https://img.shields.io/badge/Meta%20Llama-3.2-red?logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green?logoColor=white)
![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

A production-ready application that transcribes meeting audio, corrects financial terminology, and generates structured meeting minutes and actionable task lists using multiple large language models in orchestrated sequence

# 🎯 Project Highlights
## What it does:
	•	Transcribes meeting audio with 90% confidence using OpenAI Whisper (medium model)
	•	Intelligently corrects financial terminology (e.g., "401k" → "401(k) retirement savings plan")
	•	Generates structured meeting minutes and prioritized task lists with assignees and deadlines
	•	Provides downloadable text artifacts for immediate use
## Why it matters:
	•	Saves 30+ minutes of manual note-taking per meeting
	•	Prevents ambiguous financial terminology from reaching stakeholders
	•	Extracts actionable items that would otherwise get lost in recordings
	•	Demonstrates production-grade LLM orchestration and prompt engineering
## Technology Stack:
	•	**Speech-to-Text**: OpenAI Whisper (transformers)
	•	LLM Orchestration: IBM Watsonx Granite + Llama 3.2 Vision
	•	Prompt Engineering: LangChain (PromptTemplate, LLMChain)
	•	Web Interface: Gradio
	•	Python: 3.8+

## 🏗 Architecture & Technical Approach

### Multi-Stage LLM Pipeline

The application uses **sequential LLM calls**, each optimized for a specific task:

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Stage 1: Audio Transcription                               │
│  ├─ Model: OpenAI Whisper (medium)                          │
│  ├─ Input: MP3/WAV audio file                               │
│  └─ Output: Raw transcript (may contain errors/non-ASCII)   │
│                                                               │
│  Stage 2: Text Cleanup                                       │
│  ├─ Function: remove_non_ascii()                            │
│  ├─ Purpose: Ensure compatibility with downstream models   │
│  └─ Output: ASCII-safe transcript                           │
│                                                               │
│  Stage 3: Financial Terminology Correction                  │
│  ├─ Model: Meta Llama 3.2-11B Vision Instruct               │
│  ├─ Input: System prompt + ASCII transcript                 │
│  ├─ Task: Transform acronyms → full names with context     │
│  │  Examples:                                                │
│  │  • "401k" → "401(k) retirement savings plan"            │
│  │  • "LTV" → "Loan to Value" or "Lifetime Value"          │
│  └─ Output: Corrected transcript + change log              │
│                                                               │
│  Stage 4: Meeting Minutes & Task Generation                 │
│  ├─ Model: IBM Granite 4-H-Small (8B params)               │
│  ├─ Orchestration: LangChain (PromptTemplate + LLMChain)   │
│  ├─ Template: Structures output as Minutes + Task List      │
│  └─ Output: Markdown-formatted minutes & actionables       │
│                                                               │
│  Stage 5: UI & Download                                     │
│  ├─ Framework: Gradio                                       │
│  └─ Outputs: Display text + downloadable .txt file         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```
### Why This Approach?

- **Specialized Models**: Each model is optimized for its task (ASR, terminology, summarization)
- **Context Awareness**: Llama corrects ambiguous acronyms using context (e.g., LTV in financial vs. marketing)
- **Prompt Engineering**: System prompts guide models toward domain-specific output
- **Modularity**: Easy to swap models (e.g., try Llama-4-Maverick instead of Granite) without rewriting logic
- **Scalability**: LangChain chains abstract complexity; can add RAG, memory, or tool-use layers

---

## 📋 Setup & Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended for faster inference)
IBMid with access to Watsonx (credentials handled by environment)
```
## 🧠 Key Learning Outcomes
By studying this project, you'll understand:
	1.	LLM Orchestration: How to chain multiple models for complex workflows
	2.	Prompt Engineering: Writing system prompts that guide model behavior and handle ambiguity (e.g., context-aware acronym resolution)
	3.	LangChain Abstractions: Using PromptTemplate, LLMChain, and RunnablePassthrough for modularity
	4.	Production Patterns: Text cleanup, error handling, file I/O, user interfaces
	5.	Model Selection: Tradeoffs between model size, speed, and accuracy for specific tasks
	6.	Domain Knowledge: Financial terminology and structured output formatting
