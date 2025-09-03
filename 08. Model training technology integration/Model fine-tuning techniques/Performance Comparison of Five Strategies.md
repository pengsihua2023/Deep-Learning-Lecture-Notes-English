

# 📊 Comparison of Five Model Fine-Tuning Strategies

| Method            | Definition                                                                             | Mathematical Formulation                                                            | Advantages                                                   | Disadvantages                                                                         | Applicable Scenarios                                                                 |
| ----------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Prompt Tuning** | Add a **learnable soft prompt** before the input layer, freeze model params            | Input is $\[P;X]\$, optimize only \$P\$                                             | Simple, few parameters, easy to implement                    | Limited expressiveness, more stable on large models, poor performance on small models | Classification/Generation tasks with large-scale pre-trained models (>1B parameters) |
| **P-Tuning v1**   | Add soft prompt at input layer, learn prompts via **LSTM/MHA**                         | $\[g(P);X]\$, where \$g\$ is LSTM/MHA                                               | Flexible, can model complex prompts                          | Training unstable, high overhead                                                      | Downstream tasks on small models (e.g., BERT-base)                                   |
| **P-Tuning v2**   | Add prefix embeddings in **every Transformer layer**, similar to Prefix Tuning         | Concatenate prefix to attention Q/K/V: \$\text{Attn}(\[XW\_Q;P\_Q],\[XW\_K;P\_K])\$ | Strong expressiveness, performance close to full fine-tuning | More parameters than Prompt Tuning, more complex implementation                       | Various models (BERT, GPT, T5), tasks like classification, generation, extraction    |
| **LoRA**          | Insert **low-rank decomposition update** into weight matrix \$W\$: \$\Delta W = AB^T\$ | \$W' = W + AB^T\$, optimize only \$A,B\$                                            | Efficient, combinable with other methods, inference-friendly | Slightly slower convergence on some tasks                                             | Large model fine-tuning (especially generation tasks, e.g., LLaMA, GPT)              |
| **Model Pruning** | Remove unimportant weights/channels, then fine-tune remaining parameters               | \$W' = W \odot M\$, where \$M \in {0,1}^n\$                                         | Reduces model size, accelerates inference                    | Risk of performance drop, requires fine-tuning                                        | Deployment optimization (mobile/low-compute devices), model compression              |

---

# ✅ Summary

* If the goal is **minimal parameter overhead**: Prompt Tuning.
* If the model is small and needs flexible expressiveness: P-Tuning v1.
* If you want a balance between performance and parameter efficiency: P-Tuning v2.
* If you want a scalable and inference-friendly solution: LoRA.
* If the focus is on **model compression and acceleration**: Model Pruning.


