from datetime import datetime
import os
from pathlib import Path
import subprocess
from omegaconf import OmegaConf


def flatten_config(cfg):
    flat_cfg = {}
    for k in cfg:
        v = cfg[k]
        if isinstance(v, dict):
            flat_cfg.update({"-".join([k, _k]): _v for _k, _v in flatten_config(v).items()})
        else:
            flat_cfg[k] = v
    return flat_cfg


def to_carg(k, v):
    if isinstance(v, str):
        v = f'"{v}"'
    elif isinstance(v, list):
        v = f'"{str(v)}"'
    elif isinstance(v, bool):
        v = 1 if v else 0

    return f"--{k} {v}"


if __name__ == "__main__":

    path_cfg = OmegaConf.from_cli().config
    cfg = OmegaConf.load(path_cfg)

    pathtracer_cfg_dict = OmegaConf.to_container(cfg.pathtracer, resolve=True)
    pathtracer_cfg_flat = flatten_config(pathtracer_cfg_dict)
    cfg_cargs = {to_carg(k, v) for k, v in pathtracer_cfg_flat.items()}

    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_output = Path("output") / datestr
    os.makedirs(dir_output, exist_ok=False)
    OmegaConf.save(cfg, dir_output / "config.yml")

    for path_input in cfg.inputs:

        name = Path(path_input).stem

        subdir_output = dir_output / name
        os.makedirs(subdir_output, exist_ok=False)

        command = [
            cfg.paths.executable,
            *cfg_cargs,
            f"--path-gltf {path_input}",
            f"--dir-output {str(subdir_output)}",
        ]

        print("Running process:")
        print()
        print(f"{command[0]}")
        for arg in command[1:]:
            print(f"  {arg}")
        print()

        subprocess.run(
            " ".join(command),
            shell=True,
            stderr=True,
        )
