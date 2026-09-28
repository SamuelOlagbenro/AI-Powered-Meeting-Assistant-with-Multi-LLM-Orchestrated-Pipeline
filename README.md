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
Al-Powered Meeting Assistant with Multi-LLM Orchestration - Transcribes meeting audio with OpenAI Whisper, cleans terminology with Meta Llama 3.2, and generates structured minutes and task lists using IBM Granite, all orchestrated by LangChain and served through a Gradio web interface.

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
