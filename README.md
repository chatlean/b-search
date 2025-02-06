# bChatLean
The project focuses on generating mathematical proofs by employing our two proof search algorithms, **b-search** and **d-search**. The proof search algorithms we propose are simple and effective, based on the principles of breadth-first search and depth-first search. 

In this repository, we contain all codes and some results for bChatLean, which are based on b-search.

## Requirements
We utilize chatGPT and Lean 3 to support the mathematical proof. In our experiments, we used Lean version 3.42.1, the same version used in [miniF2F](https://github.com/openai/miniF2F).

First of all, we utilize ChatGPT as a model and Lean 3 to support the mathematical proof.

To interact with Lean, we utilize [LeanDojo](https://github.com/lean-dojo/LeanDojo), which easily checks time limits, and is well-organized.
Using LeanDojo requires several environment settings to facilitate the tracing and extraction of data.
We only document our specific settings; additional available settings can be found in the original LeanDojo repository.
- 3.9 <= python <= 3.10
- Set the environment variables below:
    ```
    export CONTAINER="native"
    export GITHUB_ACCESS_TOKEN=[Your GitHub access token]
    export CACHE_DIR=[Directory for cache files]
    ```

### Installation
Install the packages with the following command to run our project:
    ```
    pip install -r requirements.txt
    ```

### Data Preparation
We use LeanDojo to interact with Lean. Employing LeanDojo requires a data extraction process. For more information and a detailed example, refer to the [LeanDojo](https://github.com/lean-dojo/LeanDojo).

## Structure
Below is an outline of the main directories and files included in this project:
- `datasets/`: The datasets used in experiments.
    - `prompt_examples/`: Includes the examples file `examples.json` used in prompt.
    - `small_minif2f/`: A small part of MiniF2F for quick tests. 
- `logs/`: The directory for log files.
- `results/`: Contains the experiment outputs.
    - `AMC12_2023/`: Our experiment results for the 2023 AMC12 problems, which we newly formalized.
    - `Llemma/`: Our experiment results using [Llemma](https://arxiv.org/abs/2310.10631) as the base model instead of ChatGPT, with miniF2F as the problem set.
    - `miniF2F/`: Our main results in the paper.
    - `ProofNet/`: Our experiment results for the [ProofNet](https://github.com/zhangir-azerbayev/ProofNet) dataset.
- `scripts/`: Includes Python files to run bChatLean for a proof-generating.

### Run
`chatlean_bfs.py` is to search for a mathematical proofs with b-search. To run these scripts, use the following command lines:
```
python scripts/chatlean_bfs.py --API_key [OpenAI API key] --model [ChatGPT model name] --temperature [Temperature] --num_sample [Number of Samples] --data_path [Path for minif2f dataset] --ex_data datasets/prompt_examples/examples.json  --result_fname [Name of result file]
```
