# Glorified Autocomplete: GPT-2 Pretraining

This project is a complete, from-scratch implementation of a decoder-only Transformer language model, built entirely in PyTorch. It is designed to act as a highly complex autocomplete function, trained exclusively on the Tiny Shakespeare dataset.

This repository demonstrates the entire NLP pipeline: from data downloading and Byte-Pair Encoding (BPE) tokenization, to building the Multi-Head Self-Attention architecture, optimizing the training loop, and performing autoregressive text generation with temperature control.

## 🚀 Features

* **From-Scratch Architecture:** Implements Masked Self-Attention, Multi-Head Attention, Feed-Forward Networks, and Pre-LayerNorm Transformer Blocks without relying on high-level library abstractions.
* **BPE Tokenization:** Utilizes OpenAI's official `tiktoken` (GPT-2 encoding) to process text at the sub-word level, solving the semantic bottleneck of character-level models and massively expanding the effective context window.
* **Optimized Training:** Features an `AdamW` optimizer with weight decay, gradient clipping, and robust validation evaluation intervals.
* **Controlled Generation:** Includes an interactive generation script with adjustable **Temperature** to scale the logits and control the creativity/strictness of the model's predictions.

## 🗂️ Project Structure

* `dataset_tokenizer.py`: Automates the downloading of the Tiny Shakespeare dataset, initializes the `tiktoken` BPE encoder, and builds the PyTorch `Dataset` and `DataLoader` objects.
* `model.py`: Contains the core mathematical architecture of the Generative Pre-trained Transformer.
* `train.py`: The execution script that loads the data, initializes the model, runs the training loop, and saves the trained weights to a `.pt` file.
* `generate.py`: An interactive script that loads a pre-trained model and generates novel text based on a starting context and user-defined temperature.

## 🛠️ Installation & Setup

1. **Clone the repository** and navigate to the project folder.
2. **Install the required dependencies:**
   This project strictly requires PyTorch, the requests library (for data downloading), and tiktoken (for BPE).
   ```bash
   pip install torch requests tiktoken