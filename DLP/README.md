#  DLP

Official PyTorch implementation of [DLP: Dynamic Layerwise Pruning in Large Language Models](https://arxiv.org/abs/2505.23807)





## Results 

Perplexity results on WikiText. We produce the Uniform, OWL and DLP(Ours) with 70\% unstructured sparsity on LLaMA1, LLaMA2 models. The best performance result is indicated in bold.

| **Method**         | **Layerwise Sparsity** | **LLaMA1**  | **LLaMA1**  | **LLaMA1**  | **LLaMA2**  | **LLaMA2**  |
|--------------------|------------------------|-------------|-------------|-------------|-------------|-------------|
|                    |                        | **7B**      | **13B**     | **30B**     | **7B**      | **13B**     |
| **Dense**          | -                      | 5.68        | 5.09        | 4.10        | 5.47        | 4.88        |
| **Magnitude**      | **Uniform**            | 4.9e4       | 8.5e4       | 9.7e2       | 5.0e4       | 2.1e2       |
|                    | **OWL**                | 2.0e4       | 1.9e4       | 2.4e2       | 1.5e4       | 57.55       |
|                    | **Ours**               | **3.4e3**   | **7.6e3**   | **98.05**   | **8.7e3**   | **52.41**   |
| **SparseGPT**      | **Uniform**            | 25.38       | 18.93       | 12.87       | 27.84       | 19.38       |
|                    | **OWL**                | 19.95       | 14.02       | 10.22       | 19.71       | 15.12       |
|                    | **Ours**               | **17.76**   | **12.63**   | **9.43**    | **18.58**   | **13.30**   |
| **Wanda**          | **Uniform**            | 86.38       | 56.26       | 17.54       | 76.84       | 45.76       |
|                    | **OWL**                | 24.46       | 16.23       | 10.77       | 30.58       | 20.65       |
|                    | **Ours**               | **20.46**   | **13.65**   | **9.93**    | **22.79**   | **16.19**   |


## Installation 

Installation instructions can be found in [INSTALL.md](INSTALL.md).



## Usage

We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude_dlp`, `wanda_dlp`, `sparsegpt_dlp`.
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--eval_zero_shot`: Whether to compute accuracy on a zero-shot dataset. 
- `--save`: Specifies the directory where the result will be stored.

Below is an example command for pruning LLaMA-7B with DLP, to achieve unstructured 70% sparsity.

```
python   run.py \
    --model "Enoch/llama-7b-hf" \
    --alpha 0.15 \
    --prune_method "wanda_dlp" \
    --sparsity_ratio 0.7 \
    --sparsity_type "unstructured"
```    

zero-shot evaluation

```
python   run.py \
    --model "Enoch/llama-7b-hf" \
    --alpha 0.15 \
    --prune_method "wanda_dlp" \
    --sparsity_ratio 0.7 \
    --sparsity_type "unstructured" \
    --save_model "pruned/llama-7b-hf_sparsity_0.7_wanda_dlp" \
    --eval_zero_shot 
```   

### Acknowledgement
This repository is build upon the [OWL](https://github.com/luuyin/OWL) repositories.


### Citation

```
@misc{chen2025dlpdynamiclayerwisepruning,
      title={DLP: Dynamic Layerwise Pruning in Large Language Models}, 
      author={Yuli Chen and Bo Cheng and Jiale Han and Yingying Zhang and Yingting Li and Shuhao Zhang},
      year={2025},
      eprint={2505.23807},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23807}, 
}
```


