import os
import torch

class Saver(object):
    def __init__(self, cfg):
        self.experiment_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME)
        self.experiment_checkpoints_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.experiment_code_dir = os.path.join(self.experiment_dir, "code")
        os.makedirs(self.experiment_checkpoints_dir, exist_ok=True)
        os.makedirs(self.experiment_code_dir, exist_ok=True)
        os.system("rsync -avm --exclude='_*/' --exclude='out/' --exclude='data/' --include='*/' --include='*.py' --exclude='*' ./ " + self.experiment_code_dir)

    def save_checkpoint(self, state, is_best, filename="checkpoint-latest.pth"):
        if is_best:
            filename = "checkpoint-best.pth"
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a') as f:
                f.write(str(state["epoch"]) + ", " + str(best_pred) + "\n")
        torch.save(state, os.path.join(self.experiment_checkpoints_dir, filename))

    def save_metrics(self, epoch, metrics):
        out_dir = os.path.join(self.experiment_dir, 'metrics')

        for (m, v) in metrics.items():
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"metric_{m}.txt"), 'a') as f:
                f.write(str(epoch) + ", " + str(v) + "\n")

    def load_checkpoint(self, checkpoint_file, model, optimizer=None, device="cuda", map_location="cpu"):
        state = {}
        if checkpoint_file is not None:
            if not os.path.isfile(checkpoint_file):
                raise RuntimeError(f"=> Resume checkpoint does not exist! ({checkpoint_file})")

            checkpoint = torch.load(checkpoint_file, map_location=map_location)
            
            try:
                model.load_state_dict(checkpoint['state_dict'])
                state["start_epoch"] = checkpoint['epoch']
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(device)
                if "best_pred" in checkpoint: 
                    state["best_pred"] = checkpoint['best_pred']
            except:
                # finetuning or using some part of pretrained model
                print(f"Failed to load original model {checkpoint_file}") 
                print("    Loading only matching layers ...")
                print("    Not loading saved optimizer ...")
                model_state = model.state_dict()
                pretrained_state = { k:v for k,v in checkpoint['state_dict'].items() if k in model_state and v.size() == model_state[k].size() }
                no_match = { k:v.size() for k,v in checkpoint['state_dict'].items() if (k in model_state and v.size() != model_state[k].size()) or (k not in model_state) }
                print("    Not matched parts: \n", no_match)
                model_state.update(pretrained_state)
                model.load_state_dict(model_state, strict=False)
                update_optimizer = False
            
            print(f"=> loaded checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")
        return state

    def save_experiment_config(self, cfg):
        with open(os.path.join(self.experiment_dir, 'parameters.yaml'), 'w') as f:
            f.write(cfg.dump())


