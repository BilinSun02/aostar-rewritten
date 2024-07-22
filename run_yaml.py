import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config/benchmark", config_name="miniF2F_curriculum_test")
def go(cfg : DictConfig) -> None:
    print(cfg.datasets[0].files[0].theorems)

if __name__ == "__main__":
    go()