import hydra
from omegaconf import DictConfig, OmegaConf

from frost.launcher import launch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    launch(cfg)


if __name__ == "__main__":
    main()
