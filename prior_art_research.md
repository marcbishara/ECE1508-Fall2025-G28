# Prior art search on fine tuning LM with RLHF and PPO

### Purpose

This document will contain notes on existing research in the field with a brief description.

| Title                                                        | Link                                                         | Description and notes                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training language models to follow instructions with human feedback | [InstructGPT Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) | Paper about InstructGPT (Ouyang et al). Authors show how a much smaller model can be trained with PPO and RLHF to be much better aligned to human preference than a much larger model. Contains the useful 3 step diagram of the full process. |
| Proximal Policy Optimization (PPO) for LLMs Explained Intuitively | [video](https://youtu.be/8jtAzxUwDj0?si=nKfWaJWXZoYZfqBR) | Youtube video that goes through the explanation of the loss functions in PPO RLHF implementation in detail. |
| Fine-tuning LLMs on Human Feedback (RLHF + DPO)              | [video](https://youtu.be/bbVoDXoPrPM?si=pjCuPdrST95Y_L7u)    | Youtube video that shows a step by step example of implementing RLHF with PPO for fine tuning a model to generate youtube video titles that are more in line with creator's taste. |
| Proximal Policy Optimization Algorithms                      | [Original PPO Paper](https://arxiv.org/pdf/1707.06347)       | The original paper from 2017 in which the PPO algorithm was first proposed (Schulman et al). |
| Fine-Tuning Language Models from Human Preferences                    | [PPO for LMs](https://arxiv.org/abs/1909.08593)       | Foundational paper for using RL (and PPO in particular) to fine-tune language models (Ziegler et al). |
| Learning to summarize from human feedback                    | [Paper](https://arxiv.org/pdf/2009.01325)                    | An application of PPO on a summarization task. Human "labelling" ("comparison") in involved. |
| Secrets of RLHF in Large Language Models Part I: PPO         | [Paper](https://arxiv.org/pdf/2307.04964)                    | The article introduced an improvement on base-PPO with an extra penalty term added to the reward function. The rest parts explain a detailed RLHF process. | 
| Self-Rewarding PPO: Aligning Large Language Models with Demonstrations Only | [Paper](https://openreview.net/pdf?id=cOlHP5E3qF) | This paper proposed a self-rewarding PPO system, in which the reward function can be trained using demonstration data only (e.g. no human annotation of preference is involved). | 
| REINFORCE++: An Efficient RLHF Algorithm with Robustnessto Both Prompt and Reward Models | [download page of paper](https://www.researchgate.net/publication/387487679_REINFORCE_An_Efficient_RLHF_Algorithm_with_Robustnessto_Both_Prompt_and_Reward_Models) | This paper introduced a "global advantage normalization" step to optimize the performance of PPO in a few senarios. (This might not be that helpful to our project). |
| Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback | [Paper](https://arxiv.org/abs/2307.15217) | Outlines various challenges with different aspects of RLHF, catgorized as "fundamental" or "tractable" (Casper et al). |
| Delve into PPO: Implementation Matters for Stable RLHF | [Paper](https://openreview.net/pdf?id=rxEmiOEIFL) | Some modifications to PPO (ByteDance). |
| Supervised Fine-Tuning |[Hugging Face Docs](https://huggingface.co/learn/llm-course/chapter11/3) | Shows how to use TRL’s SFTTrainer, with steps for data, config, and training code. |
| TRL Example Overview (PPO & SFT) | [Hugging Face Docs](https://huggingface.co/docs/trl/en/example_overview) | Official Hugging Face TRL examples for Supervised Fine-Tuning (SFT) and Proximal Policy Optimization (PPO).|
|                |                |                  |

### Other
Other fine-tuning techniques such as DPO (Direct Preference Optimization) and SimPO (simlet version of DPO) use a different reward formulation. We are not considering these for this project, as they do not have an explicit reward model (i.e., do not train a reward model and then optimize a policy model to maximize that reward, as in PPO).

Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

Meng et al. SimPO: Simple Preference Optimization with a Reference-Free Reward. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)

