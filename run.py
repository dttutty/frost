import hydra
from omegaconf import DictConfig

from frost.launcher import launch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    launch(cfg)


if __name__ == "__main__":
    main()
