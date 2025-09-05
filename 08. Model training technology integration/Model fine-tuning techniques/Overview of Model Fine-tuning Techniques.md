# Overview of Fine-Tuning Techniques

## 📖 1. What is Fine-Tuning?

* **Fine-Tuning** refers to the process of taking a **pre-trained model** and further training it with downstream task data so that the model adapts to a specific task.
* Compared with **training a model from scratch**, fine-tuning usually:

  * Requires less data;
  * Converges faster;
  * Achieves better generalization performance.

Fine-tuning is the core approach of modern **Transfer Learning**, especially after the emergence of large-scale pre-trained models (e.g., BERT, GPT, ResNet, Vision Transformer), and has become a standard paradigm.



## 📖 2. Main Categories of Fine-Tuning Techniques

### (1) Full Fine-Tuning

* Idea: Load the pre-trained model → **update all parameters**.
* Pros: Flexible, best performance;
* Cons: Huge number of parameters, high computation/storage cost, not suitable for extremely large models (e.g., 10B+ parameter LLMs).



### (2) Parameter-Efficient Fine-Tuning (PEFT)

Goal: Only update a small subset of parameters to greatly reduce computation and storage cost.
Common methods:

* **Adapter Tuning**

  * Insert small bottleneck networks (down → non-linear → up) inside Transformer layers, only train the Adapter.
  * Formula: \$h' = h + W\_{up},\sigma(W\_{down} h)\$.

* **LoRA (Low-Rank Adaptation)**

  * Freeze original weights, only learn low-rank increments:
    \$\Delta W = BA\$.
  * Feature: Highly efficient, suitable for LLMs.

* **Prefix Tuning / Prompt Tuning**

  * Inject **trainable prefix vectors** into the attention mechanism without modifying model weights.
  * Commonly used in text generation tasks.

* **QLoRA**

  * On top of LoRA, quantize the large model to 4-bit, then insert LoRA, further reducing memory usage.

* **Other Extensions**

  * **LoHA**: Low-rank + Hadamard product for enhanced expressivity.
  * **LoKr**: Use Kronecker product to construct low-rank updates, further reducing parameters.



### (3) Partial Layer Freezing

* Method: Only update some layers (e.g., last few), freeze the others.
* Suitable for tasks with limited data to avoid overfitting.



### (4) Multi-task & Continual Fine-Tuning

* **Multi-task Fine-Tuning**: A single model adapts to multiple tasks simultaneously.
* **Continual Fine-Tuning**: Continue training on new tasks while avoiding catastrophic forgetting.



## 📖 3. Pros and Cons Comparison

| Method               | Pros                                      | Cons                                          | Suitable Scenarios                |
| -------------------- | ----------------------------------------- | --------------------------------------------- | --------------------------------- |
| Full Fine-Tuning     | Best performance, flexible                | Huge parameters, high storage/compute         | Small models or critical tasks    |
| Adapter              | Few parameters, task switching            | Requires modifying model structure            | Multi-task learning, NLP/NLU      |
| LoRA / QLoRA         | Tiny parameter size, low memory           | Requires choosing proper insertion            | LLM instruction tuning, chatbots  |
| Prefix/Prompt Tuning | Decoupled from model size, minimal params | Limited expressivity, may require long prefix | Text generation, dialogue systems |
| Layer Freezing       | Simple, prevents overfitting              | Limited flexibility, may underperform         | Small tasks with limited data     |


## 📖 4. Application Scenarios

* **Natural Language Processing (NLP)**

  * Text classification, QA, machine translation, dialogue systems
  * Examples: BERT fine-tuning for text classification; GPT fine-tuning for dialogue tasks

* **Computer Vision (CV)**

  * Image classification, object detection, medical image analysis
  * Example: ResNet fine-tuned for medical image recognition

* **LLM Fine-Tuning**

  * Instruction tuning
  * Alignment: RLHF, DPO
  * Industry-specific applications: finance, law, healthcare LLMs

## 📖 Summary

* **Fine-Tuning** = Adapting downstream tasks based on pre-trained models
* **Full Fine-Tuning**: Flexible but expensive
* **PEFT**: Adapter, LoRA, Prefix/Prompt Tuning, QLoRA → mainstream today
* **Key to choosing method**: task scale, data availability, memory limits, performance requirements


## 📖 Fine-Tuning Technical Roadmap

```
Fine-Tuning
│
├── Full Fine-Tuning
│   └── Update all parameters
│       ├─ Pros: flexible, good performance
│       └─ Cons: huge params, high compute cost
│
├── PEFT (Parameter-Efficient Fine-Tuning)
│   │
│   ├── Adapter
│   │   └── Insert bottleneck per layer (down→non-linear→up)
│   │
│   ├── LoRA (Low-Rank Adaptation)
│   │   └── Low-rank matrix decomposition ΔW = B A
│   │
│   ├── QLoRA
│   │   └── Quantization (4-bit) + LoRA
│   │
│   ├── Prefix Tuning
│   │   └── Add prefix key/value in attention layer
│   │
│   ├── Prompt Tuning
│   │   └── Train learnable prompt vectors
│   │
│   ├── LoHA
│   │   └── ΔW = (B A) ⊙ (D C), Hadamard enhanced
│   │
│   └── LoKr
│       └── ΔW = A ⊗ B, Kronecker product compression
│
├── Layer Freezing
│   └── Freeze part of layers, train last few
│
└── Special Forms
    ├── Multi-task Fine-Tuning
    ├── Continual Fine-Tuning
    └── Instruction Tuning / RLHF / DPO (LLM alignment)
```


## 📖 Interpretation

* **Two main branches**: full fine-tuning and parameter-efficient fine-tuning.
* **PEFT methods**: mainstream for large model fine-tuning, covering Adapter, LoRA, QLoRA, Prefix/Prompt Tuning, etc.
* **Extended methods**: LoHA, LoKr as improvements on LoRA.
* **Special forms**: multi-task, continual learning, alignment methods.



