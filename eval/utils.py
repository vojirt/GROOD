import os
import torch
import numpy as np
import tqdm
import gc
from prettytable import PrettyTable
from types import SimpleNamespace
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal, norm
from einops import rearrange
import pickle
import hashlib
from scipy.interpolate import interp1d

from eval.ood_metrics import metric_ood, compute_oscr


def get_features(grood, dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for _, (X, y) in enumerate(tqdm.tqdm(dataloader)):
            out = grood.evaluate(X.to(grood.device))
            all_features.append(out.emb.cpu())
            all_labels.append(y.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def cross_val_linprobe_C(train_features, train_labels, val_features, val_labels):
    # Perform logistic regression
    best_C = 0.05
    best_acc = -1
    print("Cross-val linear model C param ...")
    c_sweep = np.linspace(0.05, 3.0, num=100)
    
    train_features_subset = train_features
    train_labels_subset = train_labels
    # subsample training data (too slow otherwise)
    if train_features.shape[0] > 10000:
        tf = []
        tl = []
        uc = np.unique(train_labels)
        for c in uc: 
            mask = np.nonzero(train_labels == c)[0]
            mask = np.random.choice(mask, size=np.min([len(mask), np.max([int(0.1 * train_labels.shape[0] / float(len(uc))), 50])]), replace=False)
            tf.append(train_features[mask, ...])
            tl.append(train_labels[mask])
        train_features_subset = np.concatenate(tf, axis=0)
        train_labels_subset= np.concatenate(tl, axis=0)

    print(f"Using {train_features_subset.shape[0]} training data for cross-val.")

    tbar = tqdm.tqdm(c_sweep, desc='\r')
    for c in tbar:
        # classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0, n_jobs=8)
        classifier = LogisticRegression(random_state=0, C=c, max_iter=200, verbose=0, n_jobs=8)
        classifier.fit(train_features_subset, train_labels_subset)

        predictions = classifier.predict(val_features)
        accuracy = np.mean((val_labels == predictions).astype(np.float))
        if accuracy > best_acc:
            best_acc = accuracy
            best_C = c

        tbar.set_description(f"c {c:0.3f} (best - c: {best_C:0.3f}, acc: {100*best_acc:0.2f})")
        del classifier
    print(f"Best C = {best_C:0.3f} with val. accuracy {100*best_acc:0.2f}.")
    return best_C


def eval_switcher(cfg, grood, test_loader_ood, test_loader_id, inverse_class_map, evaluation_type, dontknow_prior=-1, train_set=None, val_set=None):
    if evaluation_type in ["arpl", "logits"]:
        arpl_mode = True if evaluation_type == "arpl" else False
        eval_test_results = eval_test_data_arpl(cfg, grood, test_loader_ood, test_loader_id, inverse_class_map, arpl_mode=arpl_mode)
    elif evaluation_type == "linprobe":
        # using linear probe, for CLIP
        if train_set is not None and val_set is not None:
            eval_test_results = eval_test_data_linprobe(cfg, grood, train_set, val_set, test_loader_ood, test_loader_id, inverse_class_map)
        else:
            raise RuntimeError
    elif evaluation_type == "nn":
        # using nearest-neighbour classification, for CLIP
        if train_set is not None and val_set is not None:
            eval_test_results = eval_test_data_nn(cfg, grood, train_set, val_set, test_loader_ood, test_loader_id, inverse_class_map)
        else:
            raise RuntimeError
    elif evaluation_type == "grood":
        if train_set is not None and val_set is not None:
            eval_test_results = eval_test_data_linprobeNMNquadCalib(cfg, grood, train_set, val_set, test_loader_ood, test_loader_id, inverse_class_map)
        else:
            raise RuntimeError
    else:
        raise RuntimeError

    return eval_test_results


def eval_test_data_arpl(cfg, grood, test_loader_ood, test_loader_id, inverse_class_map, arpl_mode=True):
    if arpl_mode:
        print("Using ARPL LOGITS MODE!")
    else:
        print("Using STANDARD LOGITS MODE!")

    logits = torch.empty((0, cfg.MODEL.NUM_CLASSES))
    targets = torch.empty((0,))

    with torch.no_grad():
        print("Processing ID test data")
        for _, (X, y) in enumerate(tqdm.tqdm(test_loader_id)):
            out = grood.evaluate(X.to(grood.device))
            targets = torch.concat((targets, y), dim=0)
            if arpl_mode:
                # ARPL logits
                de = (out.emb[:, None, :] - out.plane_normals[None, ...]).pow(2).mean(-1)
                batch_logits = de - out.logits
                logits = torch.concat((logits, batch_logits.cpu()), dim=0)
            else:
                # standard logits
                logits = torch.concat((logits, out.logits.cpu()), dim=0)

        print("Processing OOD test data")
        for _, (X, y) in enumerate(tqdm.tqdm(test_loader_ood)):
            out = grood.evaluate(X.to(grood.device))
            if arpl_mode:
                # ARPL logits
                de = (out.emb[:, None, :] - out.plane_normals[None, ...]).pow(2).mean(-1)
                batch_logits = de - out.logits
                logits = torch.concat((logits, batch_logits.cpu()), dim=0)
            else:
                # standard logits
                logits = torch.concat((logits, out.logits.cpu()), dim=0)

    dec_max, dec = torch.max(logits.detach(), 1)

    pred_id = dec_max[:targets.size(0)]
    pred_ood = dec_max[targets.size(0):]

    #convert from training (0, 1, ...) to dataset labels (cfg.DATASET.SELECTED_LABELS)
    decision_id = torch.zeros(targets.size(0), dtype=dec.dtype)
    decision_ood = torch.zeros(pred_ood.size(0), dtype=dec.dtype)
    for c in range(cfg.MODEL.NUM_CLASSES):
        decision_id[dec[:targets.size(0)]==c] = inverse_class_map[c]
        decision_ood[dec[targets.size(0):]==c] = inverse_class_map[c]
    
    return SimpleNamespace(
                        # MUST HAVE
                           pred_id=pred_id, 
                           pred_ood=pred_ood,
                           targets_id=targets,
                           decision_id=decision_id,
                           decision_ood=decision_ood)


def get_all_features_cached(cfg, grood, train_loader, test_loader_id, test_loader_ood):
    idd = cfg.EXPERIMENT.RESULT_DIR.find("_out")
    if idd > -1:
        exp_dir_from_out = os.path.normpath(cfg.EXPERIMENT.RESULT_DIR[idd:])
    else:
        exp_dir_from_out = os.path.normpath(cfg.EXPERIMENT.RESULT_DIR)

    cache_dir = cfg.EXPERIMENT.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    unique_hash = hashlib.md5(f"other{exp_dir_from_out}{cfg.DATASET.TRAIN}{cfg.DATASET.VAL}{cfg.DATASET.SELECTED_LABELS}".encode('utf-8')).hexdigest()  
    cache_file = os.path.join(cache_dir, unique_hash + ".pkl") 
    if os.path.isfile(cache_file):
        print("Loading train+val embeddings from cache ...")
        data = pickle.load(open(cache_file, 'rb')) 
        train_features, train_labels = data["feat"], data["labels"]
    else:
        print("Computing train+val embeddings ...")
        train_features, train_labels = get_features(grood, train_loader)
        print(f"Saving train+val embeddings to {cache_file}")
        pickle.dump({"feat":train_features, "labels":train_labels}, open(cache_file, 'wb')) 

    unique_hash = hashlib.md5(f"other{exp_dir_from_out}{cfg.DATASET.TRAIN}id{cfg.DATASET.OOD_SELECTED_LABELS}{cfg.DATASET.SELECTED_LABELS}".encode('utf-8')).hexdigest()  
    cache_file = os.path.join(cache_dir, unique_hash + ".pkl") 
    if os.path.isfile(cache_file):
        print("Loading test id embeddings from cache ...")
        data = pickle.load(open(cache_file, 'rb')) 
        test_features_id, test_labels_id = data["feat"], data["labels"]
    else:
        print("Computing test id embeddings ...")
        test_features_id, test_labels_id = get_features(grood, test_loader_id)
        print(f"Saving test id embeddings to {cache_file}")
        pickle.dump({"feat":test_features_id, "labels":test_labels_id}, open(cache_file, 'wb')) 

    unique_hash = hashlib.md5(f"other{exp_dir_from_out}{cfg.DATASET.TEST}ood{cfg.DATASET.OOD_SELECTED_LABELS}{cfg.DATASET.SELECTED_LABELS}".encode('utf-8')).hexdigest()  
    cache_file = os.path.join(cache_dir, unique_hash + ".pkl") 
    if os.path.isfile(cache_file):
        print("Loading test ood embeddings from cache ...")
        data = pickle.load(open(cache_file, 'rb')) 
        test_features_ood, test_labels_ood = data["feat"], data["labels"]
    else:
        print("Computing test ood embeddings ...")
        test_features_ood, test_labels_ood = get_features(grood, test_loader_ood)
        print(f"Saving test ood embeddings to {cache_file}")
        pickle.dump({"feat":test_features_ood, "labels":test_labels_ood}, open(cache_file, 'wb')) 

    return train_features, train_labels, test_features_id, test_labels_id, test_features_ood, test_labels_ood


def eval_test_data_nn(cfg, grood, train_set, val_set, test_loader_ood, test_loader_id, inverse_class_map):

    train_features, train_labels, test_features_id, test_labels_id, test_features_ood, test_labels_ood = get_all_features_cached(
            cfg, grood, DataLoader(ConcatDataset([train_set, val_set]), batch_size=cfg.INPUT.BATCH_SIZE),
            test_loader_id, test_loader_ood)

    pred_id_dist = np.zeros(shape=[test_features_id.shape[0], cfg.MODEL.NUM_CLASSES])
    pred_ood_dist = np.zeros(shape=[test_features_ood.shape[0], cfg.MODEL.NUM_CLASSES])
    for c in range(0, cfg.MODEL.NUM_CLASSES):
        mask = train_labels == c
        class_center = np.mean(train_features[mask, :], axis=0)
        pred_id_dist[:, c] = (-np.sqrt(np.sum(np.power(test_features_id - class_center[None, ...], 2), axis=-1)+1e-9))
        pred_ood_dist[:, c] = (-np.sqrt(np.sum(np.power(test_features_ood - class_center[None, ...], 2), axis=-1)+1e-9))

    pred_id, dec_id = torch.max(torch.from_numpy(pred_id_dist), 1)
    pred_ood, dec_ood = torch.max(torch.from_numpy(pred_ood_dist), 1)

    #convert from training (0, 1, ...) to dataset labels (cfg.DATASET.SELECTED_LABELS)
    decision_id = torch.zeros(dec_id.size(0), dtype=dec_id.dtype)
    decision_ood = torch.zeros(dec_ood.size(0), dtype=dec_ood.dtype)
    for c in range(cfg.MODEL.NUM_CLASSES):
        decision_id[dec_id==c] = inverse_class_map[c]
        decision_ood[dec_ood==c] = inverse_class_map[c]

    return SimpleNamespace(
                        # MUST HAVE
                           pred_id=pred_id, 
                           pred_ood=pred_ood,
                           targets_id=torch.from_numpy(test_labels_id).long(),
                           decision_id=decision_id,
                           decision_ood=decision_ood)


def eval_test_data_linprobe(cfg, grood, train_set, val_set, test_loader_ood, test_loader_id, inverse_class_map):

    train_features_all, train_labels_all, test_features_id, test_labels_id, test_features_ood, test_labels_ood = get_all_features_cached(
            cfg, grood, DataLoader(ConcatDataset([train_set, val_set]), batch_size=cfg.INPUT.BATCH_SIZE),
            test_loader_id, test_loader_ood)

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    for c in range(cfg.MODEL.NUM_CLASSES):
        class_idx = np.nonzero(train_labels_all==c)[0]
        split_size = int(cfg.DATASET.VAL_FRACTION*class_idx.shape[0])
        indices = np.random.RandomState(seed=cfg.SYSTEM.RNG_SEED).permutation(class_idx)
        train_features.append(train_features_all[indices[split_size:],...])
        train_labels.append(train_labels_all[indices[split_size:],...])
        val_features.append(train_features_all[indices[:split_size],...])
        val_labels.append(train_labels_all[indices[:split_size],...])

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    gc.collect()
    torch.cuda.empty_cache()
    
    best_C = cross_val_linprobe_C(train_features, train_labels, val_features, val_labels)

    print (f"Computing linear probe for C {best_C:0.3f}.")
    classifier = LogisticRegression(random_state=0, C=best_C, max_iter=1000, verbose=1, n_jobs=8)
    classifier.fit(np.concatenate([train_features, val_features], axis=0), np.concatenate([train_labels, val_labels], axis=0))
        
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html 
    # "confidence score for a sample is proportional to the signed distance of that sample to the hyperplane."
    pred_id, dec_id = torch.max(torch.from_numpy(classifier.decision_function(test_features_id)), 1)
    pred_ood, dec_ood = torch.max(torch.from_numpy(classifier.decision_function(test_features_ood)), 1)

    #convert from training (0, 1, ...) to dataset labels (cfg.DATASET.SELECTED_LABELS)
    decision_id = torch.zeros(dec_id.size(0), dtype=dec_id.dtype)
    decision_ood = torch.zeros(dec_ood.size(0), dtype=dec_ood.dtype)
    for c in range(cfg.MODEL.NUM_CLASSES):
        decision_id[dec_id==c] = inverse_class_map[c]
        decision_ood[dec_ood==c] = inverse_class_map[c]

    return SimpleNamespace(
                        # MUST HAVE
                           pred_id=pred_id, 
                           pred_ood=pred_ood,
                           targets_id=torch.from_numpy(test_labels_id).long(),
                           decision_id=decision_id,
                           decision_ood=decision_ood)


def eval_test_data_linprobeNMNquadCalib(cfg, grood, train_set, val_set, test_loader_ood, test_loader_id, inverse_class_map):

    train_features_all, train_labels_all, test_features_id, test_labels_id, test_features_ood, test_labels_ood = get_all_features_cached(
            cfg, grood, DataLoader(ConcatDataset([train_set, val_set]), batch_size=cfg.INPUT.BATCH_SIZE),
            test_loader_id, test_loader_ood)

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    for c in range(cfg.MODEL.NUM_CLASSES):
        class_idx = np.nonzero(train_labels_all==c)[0]
        split_size = int(cfg.DATASET.VAL_FRACTION*class_idx.shape[0])
        indices = np.random.RandomState(seed=cfg.SYSTEM.RNG_SEED).permutation(class_idx)
        train_features.append(train_features_all[indices[split_size:],...])
        train_labels.append(train_labels_all[indices[split_size:],...])
        val_features.append(train_features_all[indices[:split_size],...])
        val_labels.append(train_labels_all[indices[:split_size],...])

    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    gc.collect()
    torch.cuda.empty_cache()
    
    best_C = cross_val_linprobe_C(train_features, train_labels, val_features, val_labels)

    train_features = train_features_all
    train_labels = train_labels_all
    val_features = train_features_all
    val_labels = train_labels_all

    uniform_features = np.random.default_rng(42).uniform(low=np.min(val_features), high=np.max(val_features), size=(10000, 768))
    uniform_features = np.max(np.linalg.norm(val_features, axis=-1))*uniform_features / np.linalg.norm(uniform_features, axis=-1, keepdims=True)

    print(f"Computing linear probe for C {best_C:0.3f}.")
    classifier = LogisticRegression(random_state=0, C=best_C, max_iter=1000, verbose=1, n_jobs=8)
    classifier.fit(train_features, train_labels)

    logits_id = classifier.decision_function(test_features_id)
    logits_ood = classifier.decision_function(test_features_ood)
    logits_val = classifier.decision_function(val_features)
    logits_uniform = classifier.decision_function(uniform_features)

    def l2_dist(x, y):
        return 1./(1 + np.sqrt(np.sum(np.power(x - y, 2), axis=-1)))

    mean_id_d= np.zeros_like(logits_id)
    mean_ood_d= np.zeros_like(logits_ood)
    mean_val_d= np.zeros_like(logits_val)
    mean_uniform_d= np.zeros_like(logits_uniform)
    for c in range(0, cfg.MODEL.NUM_CLASSES):
        mask = train_labels == c
        class_center = np.mean(train_features[mask, :], axis=0)
        mean_id_d[:, c] = l2_dist(test_features_id, class_center[None, ...])
        mean_ood_d[:, c] = l2_dist(test_features_ood, class_center[None, ...])
        mean_val_d[:, c] = l2_dist(val_features, class_center[None, ...])
        mean_uniform_d[:, c] = l2_dist(uniform_features, class_center[None, ...])

    class_means = []
    class_covs = []
    ood_guest_means = []
    ood_guest_covs = []

    interp_fnc = []

    r = 2000
    grid_data = np.meshgrid(np.linspace(-np.max(logits_val) - 5, np.max(logits_val) + 5, num=r), np.linspace(0.0, 1, num=r))
    grid_data_rearrange = rearrange(np.stack(grid_data, axis=-1), "r c d -> (r c) d")

    simulated_data = np.stack([norm.rvs(loc=0, scale=np.quantile(logits_val, 0.9)/4.0, size=10000), 
                               norm.rvs(loc=0, scale=np.quantile(mean_val_d, 0.9)/8.0, size=10000)], axis=1)
    s_cov = np.cov(simulated_data.T)

    print("Estimating distributions parameters.")
    for c in tqdm.tqdm(range(0, cfg.MODEL.NUM_CLASSES)):
        mask = val_labels == c
        # [B, 2]
        val_space = np.stack([logits_val[mask, c], mean_val_d[mask, c]], axis=1)
        # [2]
        class_means.append(np.mean(val_space, axis=0))
        # [2, 2]
        class_covs.append(np.cov(val_space.T))
        
        ood_guest_means.append(np.array([0, 0]))
        ood_guest_covs.append(s_cov.copy())

        p_c = multivariate_normal.pdf(grid_data_rearrange, mean=class_means[-1], cov=class_covs[-1], allow_singular=True)
        p_ood = multivariate_normal.pdf(grid_data_rearrange, mean=ood_guest_means[-1], cov=ood_guest_covs[-1], allow_singular=True)
        r = p_c / p_ood
        rs_id = np.argsort(r) 
        y_val = []
        x_val = []
        for i in range(1, rs_id.shape[0]):
            if r[rs_id[i]] == r[rs_id[i-1]]:
                continue
            thr = 0.5*(r[rs_id[i]] + r[rs_id[i-1]])
            x_val.append(thr)
            if len(y_val) > 0:
                y_val.append(y_val[-1] + p_c[rs_id[i]])
            else:
                y_val.append(p_c[rs_id[i]])
        y_val = np.array(y_val) / np.sum(p_c)  
        interp_fnc.append(interp1d([0.0] + x_val + [np.Inf], [0] + y_val.tolist() + [1.0], kind="linear"))

    test_ood_pdf = np.zeros_like(mean_ood_d)
    test_id_pdf = np.zeros_like(mean_id_d)
    for c in range(0, cfg.MODEL.NUM_CLASSES):
        test_ood = np.stack([logits_ood[:, c], mean_ood_d[:, c]], axis=1)
        test_id = np.stack([logits_id[:, c], mean_id_d[:, c]], axis=1)

        p_x1_id = multivariate_normal.pdf(test_id, mean=class_means[c], cov=class_covs[c])
        p_x2_id = multivariate_normal.pdf(test_id, mean=ood_guest_means[c], cov=ood_guest_covs[c])
        r_id = p_x1_id / p_x2_id 

        p_x1_ood = multivariate_normal.pdf(test_ood, mean=class_means[c], cov=class_covs[c])
        p_x2_ood = multivariate_normal.pdf(test_ood, mean=ood_guest_means[c], cov=ood_guest_covs[c])
        r_ood = p_x1_ood / p_x2_ood 

        test_id_pdf[:, c] = interp_fnc[c](r_id) 
        test_ood_pdf[:, c] = interp_fnc[c](r_ood) 

    pred_id, dec_id = torch.max(torch.from_numpy(test_id_pdf), 1)
    pred_ood, dec_ood = torch.max(torch.from_numpy(test_ood_pdf), 1)

    #convert from training (0, 1, ...) to dataset labels (cfg.DATASET.SELECTED_LABELS)
    decision_id = torch.zeros(dec_id.size(0), dtype=dec_id.dtype)
    decision_ood = torch.zeros(dec_ood.size(0), dtype=dec_ood.dtype)
    for c in range(cfg.MODEL.NUM_CLASSES):
        decision_id[dec_id==c] = inverse_class_map[c]
        decision_ood[dec_ood==c] = inverse_class_map[c]

    return SimpleNamespace(
                        # MUST HAVE
                           pred_id=pred_id, 
                           pred_ood=pred_ood,
                           targets_id=torch.from_numpy(test_labels_id).long(),
                           decision_id=decision_id,
                           decision_ood=decision_ood)


def compute_metrics(results):
    results_ood = metric_ood(results.pred_id.numpy(), results.pred_ood.numpy(), verbose=False)['Bas']

    # OSCR, target labels only for ID data
    oscr = compute_oscr(results.pred_id, results.pred_ood, results.decision_id, results.targets_id)

    out = SimpleNamespace(TNR=results_ood["TNR"], 
                          AUROC=results_ood["AUROC"], 
                          DTACC=results_ood["DTACC"],   
                          AUIN=results_ood["AUIN"],
                          AUOUT=results_ood["AUOUT"],
                          OSCR=100*oscr,
                          ACC=100*((results.decision_id==results.targets_id).sum()/float(results.targets_id.size(0)))
                         )
    return out


def print_results(cfg, out, results, dontknown_prior, evaluation_type, ood_results_suffix=None):
    if ood_results_suffix is None:
        ood_results_suffix = "All"
        if cfg.DATASET.OOD_SELECTED_LABELS is not None:
            ood_results_suffix = str(np.min(cfg.DATASET.OOD_SELECTED_LABELS)) + "-" + str(np.max(cfg.DATASET.OOD_SELECTED_LABELS))

    ood_text = "All"
    if cfg.DATASET.OOD_SELECTED_LABELS is not None:
        ood_text = ",".join([str(i) for i in cfg.DATASET.OOD_SELECTED_LABELS])

    # basic metrics
    table = PrettyTable()
    table.vrules = 2 
    table.field_names = ["Method", "ACC", "TNR95", "FPR95", "AUROC", "DTACC", "AUIN", "AUOUT", "OSCR"]
    table.add_row([cfg.EXPERIMENT.NAME, f"{out.ACC:0.2f}", f"{out.TNR:0.2f}", f"{100-out.TNR:0.2f}", f"{out.AUROC:0.2f}", 
                   f"{out.DTACC:0.2f}", f"{out.AUIN:0.2f}", f"{out.AUOUT:0.2f}", f"{out.OSCR:0.2f}"])
    table_str = table.get_string(title=f"Results table - {evaluation_type} - (train={cfg.DATASET.TRAIN}, test={cfg.DATASET.TEST})")
    print("\n", table_str)
    with open(os.path.join(cfg.EXPERIMENT.RESULT_DIR, f"results_{evaluation_type}_{cfg.DATASET.TRAIN}_vs_{cfg.DATASET.TEST}_{ood_results_suffix}.log"), 'w') as f_obj:
        f_obj.write(table_str)
        f_obj.write("\n\n OOD_SELECTED_LABELS: ")
        f_obj.write(ood_text)

    # confusion table
    table = PrettyTable()
    table.vrules = 2 

    sorted_selected_labels = sorted(cfg.DATASET.SELECTED_LABELS)
    if hasattr(cfg.DATASET, "TRAIN_LABEL_NAMES") and cfg.DATASET.TRAIN_LABEL_NAMES is not None and len(cfg.DATASET.TRAIN_LABEL_NAMES) == cfg.MODEL.NUM_CLASSES:
        table.field_names = ["class \ decision"]+[f"{c} {cfg.DATASET.TRAIN_LABEL_NAMES[c]}" for c in sorted_selected_labels]+["don't know"]
    else:
        table.field_names = ["class \ decision"]+[f"{c}" for c in sorted_selected_labels]+["don't know"]

    targets = results.targets_id 
    dec = results.decision_id
    dec_ood = results.decision_ood

    if hasattr(results, "dontknow_decision"):
        # assign c+1 to dont known decision
        dec[results.dontknow_decision[:results.targets_id.size(0)] == 1] = -1 
        dec_ood[results.dontknow_decision[results.targets_id.size(0):] == 1] = -1 

    for c in sorted_selected_labels:
        mask = targets == c
        n = mask.sum().item()
        c_dec = dec[mask]
        row = [str(c)]
        if n > 0:
            for cc in sorted_selected_labels:
                row.append(f"{(c_dec == cc).sum().item() / n * 100:0.2f}")
            row.append(f"{(c_dec == -1).sum().item() / n * 100:0.2f}")
        else:
            for cc in range(0, cfg.MODEL.NUM_CLASSES+1):
                row.append(f"{0:0.2f}")

        table.add_row(row)


    row = ["ood classes"]
    for _, cc in enumerate(sorted_selected_labels):
        row.append(f"{(dec_ood == cc).sum().item() / float(dec_ood.size(0)) * 100:0.2f}")
    row.append(f"{(dec_ood == -1).sum().item() / float(dec_ood.size(0)) * 100:0.2f}")
    table.add_row(row)

    table_str = table.get_string(title=f"Confusion table - {evaluation_type} - [test acc. {out.ACC:0.2f}] (train={cfg.DATASET.TRAIN}, test={cfg.DATASET.TEST}), dontknown_prior={dontknown_prior:0.2f}")
    print("\n", table_str)
    with open(os.path.join(cfg.EXPERIMENT.RESULT_DIR, f"confusion_table_{evaluation_type}_p{dontknown_prior:0.2f}_{cfg.DATASET.TRAIN}_vs_{cfg.DATASET.TEST}_{ood_results_suffix}.log"), 'w') as f_obj:
        f_obj.write(table_str)
        f_obj.write("\n\n OOD_SELECTED_LABELS: ")
        f_obj.write(ood_text)

