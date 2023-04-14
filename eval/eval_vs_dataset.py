import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import Subset

from dataloaders import make_datasets
from grood import GROOD
from eval.utils import eval_switcher, print_results, compute_metrics
from helpers.logger import Logger, with_debugger


@with_debugger
def eval_vs_dataset(exp_dir, latest_checkpoint, evaluation_type, dontknow_prior, cfg_extras):
    grood = GROOD(exp_dir = exp_dir, eval_last_checkpoint=bool(latest_checkpoint))
    cfg = grood.cfg
    cfg.merge_from_list(cfg_extras)
    cfg.EXPERIMENT.RESULT_DIR = os.path.join(exp_dir, "results")

    if cfg.DATASET.SELECTED_LABELS is None:
        print("SELECTED_LABELS are empty => using all labels (0...MODEL.NUM_CLASSES)!")
        cfg.DATASET.SELECTED_LABELS = np.arange(cfg.MODEL.NUM_CLASSES).tolist()
    elif len(cfg.DATASET.SELECTED_LABELS) == 3 and cfg.DATASET.SELECTED_LABELS[0] < 0:
        print(f"SELECTED_LABELS are range => using labels from {cfg.DATASET.SELECTED_LABELS[1]} to {cfg.DATASET.SELECTED_LABELS[2]} including boundary!")
        cfg.DATASET.SELECTED_LABELS = np.arange(cfg.DATASET.SELECTED_LABELS[1], cfg.DATASET.SELECTED_LABELS[2]).tolist() + [cfg.DATASET.SELECTED_LABELS[2]]
        print(f"Total number of SELECTED_LABELS {len(cfg.DATASET.SELECTED_LABELS)}.")

    if cfg.DATASET.OOD_SELECTED_LABELS is not None and len(cfg.DATASET.OOD_SELECTED_LABELS) == 3 and cfg.DATASET.OOD_SELECTED_LABELS[0] < 0:
        print(f"OOD_SELECTED_LABELS are range => using labels from {cfg.DATASET.OOD_SELECTED_LABELS[1]} to {cfg.DATASET.OOD_SELECTED_LABELS[2]} including boundary!")
        cfg.DATASET.OOD_SELECTED_LABELS = np.arange(cfg.DATASET.OOD_SELECTED_LABELS[1], cfg.DATASET.OOD_SELECTED_LABELS[2]).tolist() + [cfg.DATASET.OOD_SELECTED_LABELS[2]]
        print(f"Total number of OOD_SELECTED_LABELS {len(cfg.DATASET.OOD_SELECTED_LABELS)}.")

    cfg.freeze()

    sys.stdout = Logger(cfg.EXPERIMENT.RESULT_DIR, mode='a', filename=f"console_{evaluation_type}.log")
    os.makedirs(cfg.EXPERIMENT.RESULT_DIR, exist_ok=True)

    print("CMD: python3", " ".join(sys.argv))

    # Define Dataloader
    train_set, val_set, test_set_ood, inverse_class_map = make_datasets(cfg)

    # if the train and test datasets are the same, set the OOD to the "unused" training classes
    ood_targets = test_set_ood.targets if hasattr(test_set_ood, "targets") else test_set_ood.labels
    if cfg.DATASET.TRAIN == cfg.DATASET.TEST:
        if cfg.DATASET.OOD_SELECTED_LABELS is None:
            ood_indices = [idx for idx, target in enumerate(ood_targets) if target not in grood.cfg.DATASET.SELECTED_LABELS]
        else:
            ood_indices = [idx for idx, target in enumerate(ood_targets) if target in grood.cfg.DATASET.OOD_SELECTED_LABELS]
        test_loader_ood = torch.utils.data.DataLoader(Subset(test_set_ood, ood_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)
    else:
        if cfg.DATASET.OOD_SELECTED_LABELS is None:
            test_loader_ood = torch.utils.data.DataLoader(test_set_ood, batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)
        else:
            ood_indices = [idx for idx, target in enumerate(ood_targets) if target in grood.cfg.DATASET.OOD_SELECTED_LABELS]
            test_loader_ood = torch.utils.data.DataLoader(Subset(test_set_ood, ood_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)

    cfg_alt = cfg.clone()
    cfg_alt.defrost()
    cfg_alt.DATASET.TEST = cfg.DATASET.TRAIN
    _, _, test_set_id, _ = make_datasets(cfg_alt)

    id_targets = test_set_id.targets if hasattr(test_set_id, "targets") else test_set_id.labels
    id_indices = [idx for idx, target in enumerate(id_targets) if target in grood.cfg.DATASET.SELECTED_LABELS]
    test_loader_id = torch.utils.data.DataLoader(Subset(test_set_id, id_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)

    eval_test_results = eval_switcher(cfg, grood, test_loader_ood, test_loader_id, inverse_class_map, 
                                      evaluation_type, dontknow_prior=dontknow_prior, train_set=train_set,
                                      val_set=val_set)

    results = compute_metrics(eval_test_results)
    print_results(cfg, results, eval_test_results, dontknow_prior, evaluation_type)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='', help="Path to directory with saved experiment")
    parser.add_argument('--dontknow_prior', type=float, default=0.01, help="Don't know prior threshold")
    parser.add_argument('--evaluation_type', type=str, help="Which evaluation to use (see utils/eval_switcher).")
    parser.add_argument('--latest_checkpoint', action='store_true', help="Flag if the latest checkpoint from the experiment should be used (otherwise best checkpoint is used)")
    args, unknown = parser.parse_known_args()

    with torch.no_grad():
        eval_vs_dataset(args.exp_dir, args.latest_checkpoint, args.evaluation_type, args.dontknow_prior, unknown)
