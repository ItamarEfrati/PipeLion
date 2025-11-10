import logging
import multiprocessing
import os

import hydra
from omegaconf import DictConfig

from src.orchestrator import Orchestrator

os.environ['NUMEXPR_MAX_THREADS'] = f'{multiprocessing.cpu_count()}'


@hydra.main(version_base='1.2', config_path=os.path.join('..', "config"), config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = Orchestrator(cfg)
    run_mode = cfg.get('run_mode', 'full')
    
    try:
        if run_mode == 'preprocess_and_train':
            logging.info("Starting full pipeline execution...")
            orchestrator.run_preprocessing()
            orchestrator.run_hyperparameter_tuning()
            logging.info("Full pipeline completed successfully!")
            
        elif run_mode == 'preprocess':
            orchestrator.run_preprocessing()
            
        elif run_mode == 'train':
            orchestrator.run_hyperparameter_tuning()
            
        elif run_mode == 'inference':
            orchestrator.run_inference()
            logging.info("Inference completed successfully.")

        elif run_mode == 'preprocess_and_inference':
            orchestrator.run_preprocessing()
            orchestrator.run_inference()
            logging.info("Preprocess and inference completed successfully.")
            
        else:
            logging.error(f"Unknown run_mode: {run_mode}")
            exit(1)
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        exit(1)

if __name__ == '__main__':
    main()

