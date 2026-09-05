# ═══════════════════════════════════════════════════════════════════
# carprice/cloud/hf_syncer.py
# ═══════════════════════════════════════════════════════════════════
# Hugging Face Hub se model push/pull karta hai
# ModelTrainer yahan se import karta hai — cloud logic alag file mein
#
# WHY ALAG FILE?
# model_trainer.py → training logic
# hf_syncer.py     → cloud sync logic
# Separation of concerns
###==============================================================

import os
import sys
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging
from carprice.constants.training_pipeline import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_MODEL_DIR
)


def push_model_to_huggingface(
    folder_path: str = HF_MODEL_DIR,
    repo_id:     str = HF_REPO_ID,
    repo_type:   str = HF_REPO_TYPE,
    private:     bool = False
) -> str:
    """
    final_model/ folder ko HuggingFace Hub pe push karta hai.

    Parameters:
        folder_path (str)  : local folder → default: "final_model/"
        repo_id     (str)  : HF repo → default: constants se
        repo_type   (str)  : "model"
        private     (bool) : False → public repo

    Returns:
        str : HF repo URL
    """
    try:
        from huggingface_hub import HfApi

        logging.info(f"Pushing to HuggingFace: {repo_id}")

        api = HfApi()

        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True
            # exist_ok=True → already exist → error nahi
        )
        logging.info(f"Repo ready: huggingface.co/{repo_id}")

        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type
        )

        url = f"https://huggingface.co/{repo_id}"
        logging.info(f"Model pushed: {url}")
        return url

    except Exception as e:
        raise CarPriceException(e, sys)


def pull_model_from_huggingface(
    repo_id:  str = HF_REPO_ID,
    repo_type: str = HF_REPO_TYPE,
    save_dir: str = HF_MODEL_DIR
) -> str:
    """
    HuggingFace se model download karta hai.
    app.py startup event mein call hoga — deployment pe.

    Parameters:
        repo_id   (str) : HF repo naam
        save_dir  (str) : local folder jahan save karna hai

    Returns:
        str : local folder path
    """
    try:
        from huggingface_hub import snapshot_download

        logging.info(f"Pulling from HuggingFace: {repo_id}")

        os.makedirs(save_dir, exist_ok=True)

        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=save_dir
        )

        logging.info(f"Model downloaded to: {local_path}")
        return local_path

    except Exception as e:
        raise CarPriceException(e, sys)