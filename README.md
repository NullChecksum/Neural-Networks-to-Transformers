# Deep Learning From Scratch: Neural Networks to Large Language Models

## Disclaimer

This repository represents my personal learning journey into deep learning and transformer architectures. My primary field is cybersecurity, not data science or machine learning research. I created these notebooks to understand the fundamental concepts needed to build custom models for my own cybersecurity applications.

The implementations here are educational and focused on conceptual clarity rather than production-grade optimization. While I have strived for technical accuracy, there may be discrepancies or areas for improvement. If you notice any issues or have suggestions for corrections or enhancements, I welcome your feedback and contributions. Learning is a collaborative process, and I am happy to refine these materials based on community input.

These notebooks served as my foundation for understanding how neural networks and language models work from first principles, enabling me to apply these concepts in my cybersecurity work.

---

## Project Overview

This project takes a ground-up approach to understanding deep learning and transformer architectures. Starting from the fundamental building block of a single artificial neuron, we progressively build complexity until we implement a complete GPT-style language model capable of generating text.

The goal is educational clarity: every component is implemented from scratch with detailed explanations of the mathematics, intuition, and code. No black boxes, no hidden abstractions.

## Contents

### 1. Neural Networks Fundamentals

The first notebook covers the foundational concepts of neural networks and deep learning:

**Chapter 1: The Artificial Neuron**
- Biological inspiration and mathematical formulation
- Weights, biases, and aggregation
- Implementation of a single neuron from scratch

**Chapter 2: Activation Functions**
- Why non-linearity is essential
- Detailed exploration of Sigmoid, Tanh, ReLU, and GELU
- Visualization and comparison of different activation functions

**Chapter 3: Forward Propagation**
- Building multi-layer networks
- Data flow through the network architecture
- Implementation of a complete feedforward pass

**Chapter 4: Loss Functions**
- Mean Squared Error for regression
- Binary and Categorical Cross-Entropy for classification
- Understanding perplexity in language modeling

**Chapter 5: Backpropagation**
- The chain rule and gradient computation
- Detailed mathematical derivation
- Implementation of backward pass through networks

**Chapter 6: Gradient Descent**
- Optimization algorithms and update rules
- Batch, stochastic, and mini-batch variants
- Training loops and convergence

This notebook establishes the core neural network concepts that underpin all modern deep learning systems.

### 2. Building a Large Language Model

The second notebook applies these fundamentals to construct a GPT-style transformer model:

**Chapter 1: Tokenization**
- Converting text to numerical representations
- Character-level and subword tokenization
- Integration with GPT-2 tokenizer

**Chapter 2: Embeddings**
- Token embeddings for semantic meaning
- Positional embeddings for sequence order
- Combining embeddings for transformer input

**Chapter 3: Self-Attention Mechanism**
- Query, Key, Value formulation
- Attention score computation and weighting
- Implementation of single-head attention

**Chapter 4: Multi-Head Attention**
- Parallel attention mechanisms
- Learning different relationship patterns
- Concatenation and output projection

**Chapter 5: Feed-Forward Networks**
- Position-wise transformations
- Expansion and compression architecture
- GELU activation for modern transformers

**Chapter 6: Transformer Block**
- Combining attention and feed-forward layers
- Residual connections for gradient flow
- Layer normalization for training stability

**Chapter 7: Complete GPT Model**
- Full architecture assembly
- Causal masking for autoregressive generation
- Model initialization and parameter counting

**Chapter 8: Training GPT**
- Next-token prediction objective
- Cross-entropy loss and perplexity metrics
- AdamW optimizer and training loops

**Chapter 9: Text Generation Strategies**
- Greedy decoding
- Temperature sampling
- Top-k and nucleus sampling

**Chapter 10: Fine-tuning for Classification**
- Adapting pretrained models for specific tasks
- Classification head architecture
- Transfer learning strategies

**Chapter 11: Instruction Fine-tuning**
- Training models to follow instructions
- Prompt formatting and response generation
- Foundation for chatbot-style applications

## Learning Path

This repository is designed to be studied sequentially:

1. Start with neural network fundamentals to understand how artificial neurons work, how they learn through backpropagation, and how gradient descent optimizes their parameters.

2. Progress to the transformer architecture, where attention mechanisms replace recurrent connections, enabling efficient parallel processing of sequences.

3. Understand how GPT combines these components into a powerful language model that can generate coherent text by predicting one token at a time.

4. Explore fine-tuning techniques that adapt the base model for specific applications, from classification tasks to instruction-following.

## Key Concepts Covered

**Neural Network Foundations:**
- Forward and backward propagation
- Gradient-based optimization
- Loss functions and training dynamics

**Transformer Architecture:**
- Self-attention mechanisms
- Multi-head attention
- Position-wise feed-forward networks
- Residual connections and layer normalization

**Language Modeling:**
- Next-token prediction
- Autoregressive generation
- Causal masking
- Temperature and sampling strategies

**Model Training:**
- Pre-training objectives
- Fine-tuning strategies
- Transfer learning
- Instruction tuning

## Implementation Philosophy

Every component is implemented from scratch using PyTorch, with extensive comments explaining:
- The mathematical formulation
- The intuition behind each design choice
- The practical implementation details

The code prioritizes clarity over efficiency. Production systems would optimize many of these implementations, but for learning purposes, we keep the code as readable and self-explanatory as possible.

## Prerequisites

To get the most from this project, you should have:
- Basic Python programming knowledge
- Understanding of linear algebra, vectors and matrices
- Familiarity with calculus, partial derivatives
- Conceptual understanding of machine learning

The notebooks build up the neural network and transformer concepts from first principles, so no prior deep learning experience is required.

## Usage

The notebooks are designed to be run sequentially. Each chapter builds on previous concepts, with code cells that demonstrate the implementations and visualizations that illustrate the ideas.

You can run the notebooks locally with PyTorch installed, or use cloud platforms like Google Colab for GPU acceleration when training larger models.

## From Neuron to GPT

This project demonstrates that modern Large Language Models, despite their complexity and scale, are built from simple, understandable components:

- A neuron computes a weighted sum and applies activation
- Layers of neurons transform data through the network
- Backpropagation computes gradients for learning
- Attention mechanisms let tokens interact dynamically
- Transformers stack these mechanisms efficiently
- GPT applies this architecture to language modeling

By implementing each piece yourself, you gain intuition for how these systems work and why they are designed the way they are.

## Contributing

Feedback, corrections, and suggestions are welcome. Please feel free to open an issue or submit a pull request if you spot any errors or have ideas for improvements.

## License

This is an educational project. Feel free to use, modify, and learn from the code.
