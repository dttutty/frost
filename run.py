import hydra
from omegaconf import DictConfig, OmegaConf

from frost.launcher import launch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    print(cfg_dict)
    launch(cfg)


if __name__ == "__main__":
    main()
