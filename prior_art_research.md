# Prior art search on fine tuning LM with RLHF and PPO

### Purpose

This document will contain notes on existing research in the field with a brief description.

| Title                                                        | Link                                                         | Description and notes                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training language models to follow instructions with human feedback | [InstructGPT Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) | Paper about InstructGPT. Authors show how a much smaller model can be trained with PPO and RLHF to be much better aligned to human preference than a much larger model. Contains the useful 3 step diagram of the full process. |
| Proximal Policy Optimization (PPO) for LLMs Explained Intuitively | [video](https://youtu.be/8jtAzxUwDj0?si=nKfWaJWXZoYZfqBR) | Youtube video that goes through the explanation of the loss functions in PPO RLHF implementation in detail. |
| Fine-tuning LLMs on Human Feedback (RLHF + DPO)              | [video](https://youtu.be/bbVoDXoPrPM?si=pjCuPdrST95Y_L7u)    | Youtube video that shows a step by step example of implementing RLHF with PPO for fine tuning a model to generate youtube video titles that are more in line with creator's taste. |
| Proximal Policy Optimization Algorithms                      | [Original PPO Paper](https://arxiv.org/pdf/1707.06347)       | The original paper from 2017 in which the PPO algorithm was first proposed. |
| Learning to summarize from human feedback                    | [Paper](https://arxiv.org/pdf/2009.01325)                    | An application of PPO on a summarization task. Human "labelling" ("comparison") in involved. |
| Secrets of RLHF in Large Language Models Part I: PPO         | [Paper](https://arxiv.org/pdf/2307.04964)                    | The article introduced an improvement on base-PPO with an extra penalty term added to the reward function. The rest parts explain a detailed RLHF process. | 
| Self-Rewarding PPO: Aligning Large Language Models with Demonstrations Only | [Paper](https://openreview.net/pdf?id=cOlHP5E3qF) | This paper proposed a self-rewarding PPO system, in which the reward function can be trained using demonstration data only (e.g. no human annotation of preference is involved). | 
| REINFORCE++: An Efficient RLHF Algorithm with Robustnessto Both Prompt and Reward Models | [download page of paper](https://www.researchgate.net/publication/387487679_REINFORCE_An_Efficient_RLHF_Algorithm_with_Robustnessto_Both_Prompt_and_Reward_Models) | This paper introduced a "global advantage normalization" step to optimize the performance of PPO in a few senarios. (This might not be that helpful to our project). |
|                |                |                  |
|                |                |                  |
|                |                |                  |
