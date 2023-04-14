import sys
import argparse
import torch
from torch.utils.data import Subset
import os

from dataloaders import make_datasets
from grood import GROOD
from eval.utils import eval_switcher, print_results, compute_metrics
from helpers.logger import Logger, with_debugger


@with_debugger
def eval_ssb(exp_dir, latest_checkpoint, evaluation_type, dontknow_prior, eval_variant, cfg_extras):
    grood = GROOD(exp_dir = exp_dir, eval_last_checkpoint=bool(latest_checkpoint))
    cfg = grood.cfg
    cfg.merge_from_list(cfg_extras)
    cfg.EXPERIMENT.RESULT_DIR = os.path.join(exp_dir, "results")

    cfg.freeze()

    sys.stdout = Logger(cfg.EXPERIMENT.RESULT_DIR, mode='a', filename=f"console_{evaluation_type}.log")
    os.makedirs(cfg.EXPERIMENT.RESULT_DIR, exist_ok=True)

    print("CMD: python3", " ".join(sys.argv))

    # prepare dataloaders
    train_set, val_set, test_set, inverse_class_map = make_datasets(cfg)

    id_targets = test_set.targets
    id_indices = [idx for idx, target in enumerate(id_targets) if target in cfg.DATASET.SELECTED_LABELS]
    test_loader_id = torch.utils.data.DataLoader(Subset(test_set, id_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)
    print(f"ID dataset: {cfg.DATASET.TRAIN}")

    ood_targets = test_set.targets
    ood_indices = [idx for idx, target in enumerate(ood_targets) if target in cfg.DATASET.OOD_SELECTED_LABELS]
    test_loader_ood = torch.utils.data.DataLoader(Subset(test_set, ood_indices), batch_size=cfg.INPUT.BATCH_SIZE, shuffle=False)
    print(f"OOD dataset: {cfg.DATASET.TEST}, variant {eval_variant}")

    eval_test_results = eval_switcher(cfg, grood, test_loader_ood, test_loader_id, inverse_class_map, 
                                      evaluation_type, dontknow_prior=dontknow_prior, train_set=train_set,
                                      val_set=val_set)

    results = compute_metrics(eval_test_results)
    print_results(cfg, results, eval_test_results, dontknow_prior, evaluation_type, ood_results_suffix=eval_variant)

    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='', help="Path to directory with saved experiment")
    parser.add_argument('--dontknow_prior', type=float, default=0.01, help="Don't know prior threshold")
    parser.add_argument('--evaluation_type', type=str, help="Which evaluation to use (see utils/eval_switcher).")
    parser.add_argument('--eval_variant', choices=["Easy", "Medium", "Hard"], 
                        type=str, default="Easy", help="Benchmark difficulty")
    parser.add_argument('--latest_checkpoint', action='store_true', help="Flag if the latest checkpoint from the experiment should be used (otherwise best checkpoint is used)")
    args, unknown = parser.parse_known_args()

    with torch.no_grad():
        eval_ssb(args.exp_dir, args.latest_checkpoint, args.evaluation_type, args.dontknow_prior, args.eval_variant, unknown)

