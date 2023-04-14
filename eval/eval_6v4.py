import os
import sys
import argparse
import torch
import glob
from torch.utils.data import Subset
import numpy as np
from prettytable import PrettyTable

from helpers.logger import Logger, with_debugger
from grood import get_experiment_cfg, GROOD
from dataloaders import make_datasets
from eval.utils import print_results, eval_switcher, compute_metrics


@with_debugger
def eval_6v4(exp_dir, latest_checkpoint, evaluation_type, dontknow_prior, cfg_extras):
   
    exp_dir_list = sorted([d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d) and os.path.isfile(os.path.join(d, "parameters.yaml"))])
    if len(exp_dir_list) != 5:
       raise RuntimeError(f"Expected 5 experiments inside {exp_dir}, found {len(exp_dir_list)}! \n {exp_dir_list}") 

    cfg = get_experiment_cfg(exp_dir_list[0])
    cfg.merge_from_list(cfg_extras)
    cfg.EXPERIMENT.RESULT_DIR = os.path.join(exp_dir, "results")
    cfg.DATASET.TEST = cfg.DATASET.TRAIN

    sys.stdout = Logger(cfg.EXPERIMENT.RESULT_DIR, mode='a', filename=f"console_{evaluation_type}.log")
    os.makedirs(cfg.EXPERIMENT.RESULT_DIR, exist_ok=True)

    print("CMD: python3", " ".join(sys.argv))

    # results table
    table = PrettyTable()
    table.vrules = 2 
    table.field_names = ["Method", "ACC", "TNR", "AUROC", "DTACC", "AUIN", "AUOUT", "OSCR"]

    TNRs = [] 
    AUROCs = []
    DTACCs = []
    AUINs = []
    AUOUTs = []
    OSCRs = []
    ACCs = []
    for exp_dir_run in exp_dir_list:
        grood = GROOD(exp_dir = exp_dir_run, eval_last_checkpoint=bool(latest_checkpoint))

        cfg = get_experiment_cfg(exp_dir_run)
        cfg.merge_from_list(cfg_extras)
        cfg.EXPERIMENT.RESULT_DIR = os.path.join(exp_dir, "results")

        train_set, val_set, test_set, inverse_class_map = make_datasets(cfg)
             
        targets = test_set.targets if hasattr(test_set, "targets") else test_set.labels
        if cfg.DATASET.TEST == cfg.DATASET.TRAIN:
            id_indices = [idx for idx, target in enumerate(targets) if target in cfg.DATASET.SELECTED_LABELS]
            ood_indices = [idx for idx, target in enumerate(targets) if target not in cfg.DATASET.SELECTED_LABELS]

            # Sanity checks
            id_mask = np.zeros(len(targets), dtype=int)
            id_mask[id_indices] = 1 
            ood_mask = np.zeros(len(targets), dtype=int)
            ood_mask[ood_indices] = 1 
            assert np.sum(id_mask*ood_mask) == 0, "Data selection error, there is an overlap between ID and OOD classes!"
            assert np.sum((id_mask+ood_mask) > 0) == len(targets), "Data selection error, not all data were selected!"

            test_loader_ood = torch.utils.data.DataLoader(Subset(test_set, ood_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)
            test_loader_id = torch.utils.data.DataLoader(Subset(test_set, id_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)
        else:
            test_set_sub = test_set
            if cfg.DATASET.OOD_SELECTED_LABELS is not None:
                ood_indices = [idx for idx, target in enumerate(targets) if target in cfg.DATASET.OOD_SELECTED_LABELS]
                test_set_sub = Subset(test_set, ood_indices)
            
            test_loader_ood = torch.utils.data.DataLoader(test_set_sub, batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)

            cfg_alt = cfg.clone()
            cfg_alt.defrost()
            cfg_alt.DATASET.TEST = cfg.DATASET.TRAIN
            _, _, test_set_id, _ = make_datasets(cfg_alt)

            id_targets = test_set_id.targets if hasattr(test_set_id, "targets") else test_set_id.labels
            id_indices = [idx for idx, target in enumerate(id_targets) if target in cfg_alt.DATASET.SELECTED_LABELS]
            test_loader_id = torch.utils.data.DataLoader(Subset(test_set_id, id_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)

        eval_test_results = eval_switcher(cfg, grood, test_loader_ood, test_loader_id, inverse_class_map, 
                                          evaluation_type, dontknow_prior=dontknow_prior, train_set=train_set,
                                          val_set=val_set)
        out = compute_metrics(eval_test_results)
        print_results(cfg, out, eval_test_results, dontknow_prior, evaluation_type)

        TNRs.append(out.TNR)
        AUROCs.append(out.AUROC)
        DTACCs.append(out.DTACC)
        AUINs.append(out.AUIN)
        AUOUTs.append(out.AUOUT)
        OSCRs.append(out.OSCR)
        ACCs.append(out.ACC)

        table.add_row([cfg.EXPERIMENT.NAME,f"{out.ACC:0.2f}", f"{out.TNR:0.2f}", f"{out.AUROC:0.2f}", 
                       f"{out.DTACC:0.2f}", f"{out.AUIN:0.2f}", f"{out.AUOUT:0.2f}", f"{out.OSCR:0.2f}"])

    table.add_row(["averaged",
                   f"{np.mean(ACCs):0.2f}±{np.std(ACCs):0.2f}", 
                   f"{np.mean(TNRs):0.2f}±{np.std(TNRs):0.2f}", 
                   f"{np.mean(AUROCs):0.2f}±{np.std(AUROCs):0.2f}", 
                   f"{np.mean(DTACCs):0.2f}±{np.std(DTACCs):0.2f}", 
                   f"{np.mean(AUINs):0.2f}±{np.std(AUINs):0.2f}", 
                   f"{np.mean(AUOUTs):0.2f}±{np.std(AUOUTs):0.2f}",
                   f"{np.mean(OSCRs):0.2f}±{np.std(OSCRs):0.2f}"])
    table_str = table.get_string(title=f"Results table (train={cfg.DATASET.TRAIN}, test={cfg.DATASET.TEST})")
    lines = table_str.splitlines()
    table_str = "\n".join(lines[:-2]) + "\n" + lines[-1] + "\n" + lines[-2] + "\n" + lines[-1]
    print("\n", table_str)

    with open(os.path.join(cfg.EXPERIMENT.RESULT_DIR, f"results_{cfg.DATASET.TRAIN}_{latest_checkpoint}_p{dontknow_prior:0.2f}_{evaluation_type}_5run_6v4.log"), 'w') as f_obj:
        f_obj.write(table_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='', help="Path to directory with saved experiment")
    parser.add_argument('--dontknow_prior', type=float, default=0.01, help="Don't know prior threshold")
    parser.add_argument('--evaluation_type', type=str, help="Which evaluation to use (see utils/eval_switcher).")
    parser.add_argument('--latest_checkpoint', action='store_true', help="Flag if the latest checkpoint from the experiment should be used (otherwise best checkpoint is used)")
    args, unknown = parser.parse_known_args()

    with torch.no_grad():
        eval_6v4(args.exp_dir, args.latest_checkpoint, args.evaluation_type, args.dontknow_prior, unknown)
