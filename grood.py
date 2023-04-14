import os
import sys
import torch
import importlib


class GROOD():
    def __init__(self, exp_dir, eval_last_checkpoint=False) -> None:
        self.exp_dir = exp_dir
        self.code_dir = os.path.join(self.exp_dir, "code")
        if not os.path.isdir(self.code_dir):
            print("Using git repository as the code directory")
            self.code_dir = "./"

        self.eval_last_checkpoint = eval_last_checkpoint

        cfg_local = get_experiment_cfg(self.exp_dir)
        cfg_local.EXPERIMENT.OUT_DIR = os.path.abspath(os.path.join(exp_dir, '..'))

        checkpoint_name = "checkpoint-best.pth"
        if self.eval_last_checkpoint:
            checkpoint_name = "checkpoint-latest.pth"
        if os.path.isfile(os.path.join(self.exp_dir, "checkpoints", checkpoint_name)):
            cfg_local.EXPERIMENT.RESUME_CHECKPOINT = os.path.join(self.exp_dir, "checkpoints", checkpoint_name)
        else:
            raise RuntimeError(f"Experiment dir does not contain {checkpoint_name}!")

        # CUDA
        if not torch.cuda.is_available():
            print ("GPU is disabled")
            cfg_local.SYSTEM.USE_GPU = False

        self.cfg = cfg_local

        self.device = torch.device("cuda" if cfg_local.SYSTEM.USE_GPU else "cpu")

        # define the network
        sys.path.insert(0, self.code_dir)
        kwargs = {'cfg': cfg_local}
        spec = importlib.util.spec_from_file_location(cfg_local.MODEL.FILENAME, os.path.join(self.code_dir, "net", "models", cfg_local.MODEL.FILENAME + ".py"))
        model_module = spec.loader.load_module()
        print (self.code_dir, model_module)
        self.model = getattr(model_module, cfg_local.MODEL.NET)(**kwargs)
        sys.path = sys.path[1:]

        # load the model paraters
        if cfg_local.EXPERIMENT.RESUME_CHECKPOINT is not None:
            if not os.path.isfile(cfg_local.EXPERIMENT.RESUME_CHECKPOINT):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT))
            checkpoint = torch.load(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, map_location="cpu")
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg_local.EXPERIMENT.RESUME_CHECKPOINT, checkpoint['epoch']))
            del checkpoint
        else:
            raise RuntimeError("=> model checkpoint has to be provided for testing!")

        # Using cuda
        self.model.to(self.device)
        self.model.eval()

        # clean-up imported sys paths
        to_del = []
        for k, v in sys.modules.items():
            if k[:3] == "net":
                to_del.append(k)
        for k in to_del:
            del sys.modules[k]

    def evaluate(self, input):
        with torch.no_grad():
            return self.model(input)


def get_experiment_cfg(exp_dir):
    code_dir = os.path.join(exp_dir, "code")
    if not os.path.isdir(code_dir):
        print("Using git repository as the code directory")
        code_dir = "./"
    #from config import get_cfg_defaults
    config_module = importlib.util.spec_from_file_location("get_cfg_defaults", os.path.join(code_dir, "config", "defaults.py")).loader.load_module()
    cfg_fnc = getattr(config_module, "get_cfg_defaults")
    cfg_local = cfg_fnc()

    # read the experiment parameters
    if os.path.isfile(os.path.join(exp_dir, "parameters.yaml")):
        with open(os.path.join(exp_dir, "parameters.yaml"), 'r') as f:
            cc = cfg_local._load_cfg_from_yaml_str(f)
        cfg_local.merge_from_file(os.path.join(exp_dir, "parameters.yaml"))
        cfg_local.EXPERIMENT.NAME = cc.EXPERIMENT.NAME
    else:
        raise RuntimeError(f"Experiment directory does not contain parameters.yaml: {exp_dir}")
    return cfg_local


def main():
    # example of use
    model = GROOD(exp_dir = "./_out/experiments/20220719_152736_769664/")
    out = model.evaluate(torch.rand(64, 1, 28, 28).to("cuda"))
    print (out)


if __name__ == "__main__":
    main()
