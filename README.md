
# Machine Learning and  Language Models

This repository contains the code and resources for the machine learning focusing on language models. 
The project explores the capabilities of GPT-2 and Llama models through pre-training, fine-tuning, and prompting techniques.
(please refer to the `report.pdf` for more details on the project).

## Table of Contents

- [Part I: Pre-training GPT-2](#part-i-pre-training-gpt-2)
- [Part II: Instruction Fine-tuning GPT-2](#part-ii-instruction-fine-tuning-gpt-2)
- [Part III: Chain-of-Thoughts Prompting](#part-iii-chain-of-thoughts-prompting)
- [References](#references)


## Part I: Pre-training GPT-2

In this section, we pre-train the GPT-2 model on a Shakespearean text corpus to generate text in a similar style. 
The codes are missing for this part, but you can refer to the `report.pdf` for details on the training and validatin loss, 
    as well as the generated text samples.

- The training and validation loss curves show the model's learning progress (for 5000 steps of iterations).

- The generated text samples demonstrate the model's ability to mimic Shakespearean language, 
  although some grammatical inconsistencies are present due to character-level training.

## Part II: Instruction Fine-tuning GPT-2

We fine-tune the pre-trained GPT-2 model using the Alpaca-GPT4 dataset to enhance its instruction-following capabilities.

**Alpaca-GPT4 Dataset**: 
The dataset contains 52k instruction-following examples with 1.5M tokens, 
    designed to evaluate the model's ability to follow human instructions.

**Instruction Tuning Pipeline**:
The fine-tuning pipeline uses a specific template for tokenizing instruction-following data

```
### Instruction: <instruction text here>
### Input: <input text here>
### Response: <response text here>
```

**Memory Efficient Optimization**:
We experimented with different optimization techniques to reduce memory consumption and computational cost

- **AdamW**: a variant of Adam optimizer with weight decay regularization 
- **Stochastic Gradient Descent (SGD)**: update model using random mini-b (low memory but slower to converge)
- **Low-rank Adaptation (LoRA)**: freezes most model weights and updates only small matrices of parameters (memory-efficient)
- **Block Adam (BAdam)**: uses block-diagonal approximation of the Fisher information matrix (memory-efficient and speed-up convergence)

The fine-tuned models show significant improvements in instruction-following capabilities,
    with higher relevance and fluency in generated responses.

## Part III: Chain-of-Thoughts Prompting

We evaluate the effectiveness of CoT prompts on mathematical benchmarks using the Llama model,
    using the following datasets for evaluation:

- **GSM8K**: 8.5k grade-school math word problems (tests basic arithmetic operations)
- **NumGLUE**: math tasks like arithmetic and logic (tests algebraic reasoning)
- **StimuIEq**: math equation problems (tests equation-solving)
- **SVAMP**: complex math word problems (tests problem-solving)

**CoT Prompting Strategy and Results**:

The CoT prompts are designed to improve the model's reasoning capabilities by providing detailed step-by-step instructions. 
We evaluate the model's performance with different numbers of CoT examples (0-shot, 2-shot, and 4-shot).

The use of CoT prompts generally improves the model's performance, with 4-shot prompts yielding the best results for most datasets.

## References

1. Tom Brown et al. "Language models are few-shot learners". In: _Advances in Neural Information Processing Systems_. Vol. 33. 2020, pp. 1877-1901.
2. Karl Cobbe et al. "Training verifiers to solve math word problems". In: _arXiv preprint arXiv:2110.14168_ (2021).
3. Abhimanyu Dubey et al. "The LLaMA 3 herd of models". In: _arXiv preprint arXiv:2407.21783_ (2024).
4. Diederik P Kingma and Max Welling. "A method for stochastic optimization". In: _arXiv preprint arXiv:1412.6980_ (2014).
5. Rik Koncel-Kedziorski et al. "Mavps: A math word problem repository". In: _Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_. 2016, pp. 1152-1157.
6. Swaroop Mishra et al. "Numglue: A suite of fundamental yet challenging mathematical reasoning tasks". In: _arXiv preprint arXiv:2204.05660_ (2022).
7. Arkil Patel, Satwik Bhattamishra, and Navin Goyal. "Are nlp models really able to solve simple math word problems?" In: _arXiv preprint arXiv:2103.07191_ (2021).
8. Alec Radford et al. "Language models are unsupervised multitask learners". In: _OpenAI blog_ 1.8 (2019), p. 9.
9. Hugo Touvron et al. "LLaMA 2: Open foundation and fine-tuned chat models". In: _arXiv preprint arXiv:2307.09288_ (2023).
10. A Vaswani. "Attention is all you need". In: _Advances in Neural Information Processing Systems_. 2017.


