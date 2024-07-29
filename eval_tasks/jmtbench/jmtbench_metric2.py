import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools :import islice
from transformers import AutoTokenizer

import wandb
from wandb.sdk.wandb_run import Run
import os
import sys
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from llm_jp_eval.evaluator import evaluate
from mtbench_eval import mtbench_evaluate
from config_singleton import WandbConfigSingleton
from cleanup import cleanup_gpu

def jmtbench_eval():
    # Configuration loading
    if os.path.exists("configs/config.yaml"):
        cfg = OmegaConf.load("configs/config.yaml")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
    else:
        # Provide default settings in case config.yaml does not exist
        cfg_dict = {
            'wandb': {
                'entity': 'default_entity',
                'project': 'default_project',
                'run_name': 'default_run_name'
            }
        }

    # W&B setup and artifact handling
    wandb.login()
    run = wandb.init(
        entity=cfg_dict['wandb']['entity'],
        project=cfg_dict['wandb']['project'],
        name=cfg_dict['wandb']['run_name'],
        config=cfg_dict,
        job_type="evaluation",
    )

    # Initialize the WandbConfigSingleton
    WandbConfigSingleton.initialize(run, wandb.Table(dataframe=pd.DataFrame()))
    cfg = WandbConfigSingleton.get_instance().config

    # Save configuration as artifact
    if cfg.wandb.log:
        if os.path.exists("configs/config.yaml"):
            artifact_config_path = "configs/config.yaml"
        else:
            # If "configs/config.yaml" does not exist, write the contents of run.config as a YAML configuration string
            instance = WandbConfigSingleton.get_instance()
            assert isinstance(instance.config, DictConfig), "instance.config must be a DictConfig"
            with open("configs/config.yaml", 'w') as f:
                f.write(OmegaConf.to_yaml(instance.config))
            artifact_config_path = "configs/config.yaml"

        artifact = wandb.Artifact('config', type='config')
        artifact.add_file(artifact_config_path)
        run.log_artifact(artifact)

    # 2. mt-bench evaluation
    result = mtbench_evaluate()
    # cleanup_gpu()

    # Logging results to W&B
    if cfg.wandb.log and run is not None:
        instance = WandbConfigSingleton.get_instance()
        run.log({
            "leaderboard_table": instance.table
        })
        run.finish()

    # 評価結果
    return result

#スコアを計算して返す
def process_results():
    #ret = evaluate(results, doc, None, None)  # reference_answer と judge_prompt は evaluate 内で取得
    #try:
    #    score = sum(float(ret[f'text{i}']) for i in range(1, 4)) / 3.0
    #except ValueError:
    #    print("Warning: Could not convert score to float. Using default score of 1.")
    #    score = 1.0
    #print(f"avg: {score}, score1: {ret['text1']}, score2: {ret['text2']}, score3: {ret['text3']}")
   
    ret = jmtbench_eval()
    print(ret)
    score = ret
 
    results = {
        "acc": score,
    }
    return results
