from datetime import datetime
import os
from pathlib import Path
import subprocess
from omegaconf import OmegaConf

template_launch_json = """
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/bin/pathtracer",
            "args": [%PATHTRACER_ARGS%],
            "cwd": "${workspaceFolder}",
            "onAPIError": "stop",
            "preLaunchTask": "debug_prelaunch",
        },
    ],
}
"""


def create_hacky_launch_json(args):
    args = [arg.replace('"', "") for arg in args]  # remove ticks
    arg_list = ", ".join([f'"{arg}"' for arg in args])  # put inside new ticks and chain
    file = template_launch_json.replace("%PATHTRACER_ARGS%", arg_list)
    with open(".vscode/launch.json", "w") as f:
        f.write(file)


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

    return [f"--{k}", str(v)]


if __name__ == "__main__":

    cli = OmegaConf.from_cli()
    path_cfg = cli.config
    cfg = OmegaConf.load(path_cfg)

    pathtracer_cfg_dict = OmegaConf.to_container(cfg.pathtracer, resolve=True)
    pathtracer_cfg_flat = flatten_config(pathtracer_cfg_dict)
    cfg_cargs = []
    for k, v in pathtracer_cfg_flat.items():
        cfg_cargs.extend(to_carg(k, v))

    datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_output = Path("output") / datestr
    os.makedirs(dir_output, exist_ok=False)
    OmegaConf.save(cfg, dir_output / "config.yml")

    for path_input in cfg.inputs:

        name = Path(path_input).stem
        subdir_output = dir_output / name
        os.makedirs(subdir_output, exist_ok=False)

        input_cargs = []

        all_args = [
            *cfg_cargs,
            "--path-gltf",
            str(path_input),
            "--dir-output",
            str(subdir_output),
        ]

        if cli.make_runfile:
            create_hacky_launch_json(all_args)

        else:

            command = [cfg.paths.executable, *all_args]

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
