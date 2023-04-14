# Calibrated Out-of-Distribution Detection with a Generic Representation (GROOD)
PyTorch implementation of our work on Out-of-Distribution (OOD) image detection using a generic pre-trained models.

**\[[Paper](https://arxiv.org/abs/2303.13148)\]**


> **Abstract:** 
>*Out-of-distribution detection is a common issue in deploying vision models in
>practice and solving it is an essential building block in safety critical
>applications. Existing OOD detection solutions focus on improving the OOD
>robustness of a classification model trained exclusively on in-distribution
>(ID) data. In this work, we take a different approach and propose to leverage
>generic pre-trained representations. We first investigate the behaviour of
>simple classifiers built on top of such representations and show striking
>performance gains compared to the ID trained representations. We propose
>a novel OOD method, called GROOD, that achieves excellent performance,
>predicated by the use of a good generic representation. Only a trivial training
>process is required for adapting GROOD to a particular problem. The method is
>simple, general, efficient, calibrated and with only a few hyper-parameters.
>The method achieves state-of-the-art performance on a number of OOD benchmarks,
>reaching near perfect performance on several of them.*


# Pre-trained Representation models
In the arXiv paper we used [CLIP](https://github.com/openai/CLIP). It needs to be installed before running the training (follow
the link for the installation instructions). However, other open-source models can be used, such as
[OpenCLIP](https://github.com/mlfoundations/open_clip) or
[DIHT](https://github.com/facebookresearch/diht).

# Datasets preparation
You need to download manually these datasets:
- LSUN from [LSUN webpage](http://www.yf.io/p/lsun)
- texture, places365 and inaturalist are from [OpenOOD](https://github.com/Jingkang50/OpenOOD) (clone the repo to `./_data/OpenOOD/`; `cd` to it; and use the `OpenOOD/scripts/download/dowanload.sh` script)
- ssb_cub, ssb_scars and ssb_aircraft are from [SemanticShiftBenchmark](https://github.com/sgvaze/osr_closed_set_all_you_need) (put them into the e.g. `./_data/SemanticShiftBenchmark/` + `FGVC-Aircraft`, ...). You will need to run the scripts from `dataloaders/builders/ssb` to convert the data to the format accepted by our data loaders.
- The DomainNet dataset M3SDA, i.e. "clipart", "infograph", "painting", "quickdraw", "real" and "sketch", was obtained from the [M3SDA webpages](http://ai.bu.edu/M3SDA/) (put them into the `./_data/M3SDA/` directory)

Other datasets will be downloaded automatically from `torchvision` when used for the first time (CIFAR10, SVHN, CIFAR100, TinyImageNet, MNIST).

# Experiments
To generate the results in the paper run the following scripts: 
(by default, all experiments and results will be saved in `./_out/` directory. It can be changed in the respective script files.)

First add the repository to the python path (execute from the repository root dir):
```sh
export PYTHONPATH=`pwd`
```

**Table 2** - semantic shift only.
```sh
./eval/scripts/run_all_6v4.sh <my_method> <config_file> <gpu_id> <eval_type>
```

where:
- `<my_method>` is the method name and the results will be in the directory with the same name
- `<config_file>` config file from the `./config/` directory, e.g. `clip_L14.yaml` or it can be `None` for evaluation only (skipping training of models)
- `<gpu_id>` which gpu to use (e.g. 0)
- `<eval_type>` what evaluation type to use, e.g. grood or linprobe (for linear probe only) 
  Available evaluation types are in `./eval/utils.py` at `eval_switcher` function.

**Table 3** - mixed semantic and domain shifts
first train the model:
```sh
CUDA_VISIBLE_DEVICES=0 python train.py --config ./config/<config_file> EXPERIMENT.NAME <my_method>
```

then evaluate:
```sh
./eval/scripts/run_all_ood.sh ./_out/experiments/<my_method> <gpu_id> <eval_type>
```

**Table 4** - semantic shift (Real-B column) and domain shift only (next four columns)
```sh
./eval/scripts/run_shift_all.sh <my_method> <config_file> <gpu_id> <eval_type>"
```

**Table 5** - fine-grained semantic shift
```sh
./eval/scripts/run_all_ssb.sh <my_method> <config_file> <gpu_id> <eval_type> <skip_trainig>
```
where:
- `<skip_trainig>` set to "no" to run training + evaluation or to anything else for evaluation only

# Plots and examples in jupyter notebooks
See the notebooks in `./jupyter/` folder for the following visualizations:

**Figure 1** - Complementarity of the LP and NM classifiers

**Figure 3** - Visualization of the GROOD decision making wrt. ID-data


# Bibtex 
If you use this work please cite:

```latex
@misc{Vojir_2023_arXiv,
      title={{Calibrated Out-of-Distribution Detection with a Generic Representation}}, 
      author={Tomas Vojir and Jan Sochman and Rahaf Aljundi and Jiri Matas},
      year={2023},
      eprint={2303.13148},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Licence
Copyright (c) 2021 Toyota Motor Europe<br>
Patent Pending. All rights reserved.

This work is licensed under a [Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International
License](https://creativecommons.org/licenses/by-nc/4.0/)

