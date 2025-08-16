# CHNS: Clustering-based hard negative sampling for supervised contrastive speaker verification

This repository contains the training source code for experiments from the Interspeech 2025 publication "Clustering-based hard negative sampling for supervised contrastive speaker verification"

<p align="center">
  <a href="https://arxiv.org/abs/2507.17540" style="font-size:20px;">Full Paper</a>
</p>

<p align="center">
  <img src="resources/media/CHNS_Interspeech_poster.jpg" alt="Interspeech 2025 poster" width="300">
</p>

## Updates

**16.08.2025**: Added dataset preparation guide to readme.  
**12.08.2025**: Added training configs and poster.  
**04.07.2025**: Release a WIP version of the repo. No training configs included.

## Setup

In the root directory of this repository create a virtual environment with python 3.10.

```bash
virtualenv venv -p python3.10
```

and activate it

```bash
source venv/bin/activate
```

Install required packages. Feel free to modify the `requirements.txt` so that it matches your cuda version.

```bash
pip install -r requirements.txt
```

After trat run:

```bash
pip install -e .
```

## VoxCeleb dataset preparation

1. Go [here](https://mm.kaist.ac.kr/datasets/voxceleb/) to download the VocCeleb 1 and 2 audio files. 

   Download all of the archive parts, concatenate and unzip them in a directory of your choice. In our experiments, we use Vox1 dev, Vox1 test and Vox2 dev.

2. Create a directory for all VoxCeleb audio data in a location of your choice:

   ```sh
   mkdir VoxCeleb
   ```


2. Convert the VocCeleb2 `.aac` files to 16 kHz mono `.wav` files using a script like [this](https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830) or similar. This process often takes a long time.

3. Create helper mapping files: `spk2utt` and `utt2spk` for each dataset (Vox1 dev, Vox1 test, Vox2 dev). Run the following command with the root diretory that contains individual speaker directories as first argument:

   ```
   vox2_dev_wav/
   └── wav/        ← First argument
       ├── id00001/
       │   └── …
       ├── id00002/
       │   └── …
       ├── id00003/
       │   └── …
       └── …
   ```

   You can provide an optional output dir as second argument.    Otherwise, the `spk2utt` and `utt2spk` files will be saved do the    dir specified in the first argument.

   ```sh
   ./scripts/make_spk2utt_and_utt2spk.sh path/to/vox2_dev_wav/wav    optional/output/dir
   ```

## Train

This project uses `lightning` and specifically `LightningCLI` for configuration management. Each experiment setup should be defined in a separate `.yaml` file. The arguments can be overridden from the command line.

To run the training script of the base SupCon model run:

```
python run_train.py --config configs/supcon.yaml
```

Same for the supervised AAMSoftmax model:

```
python run_train.py --config configs/aamsoftmax.yaml
```

Remember to update your train data paths in the config files.

After the base model has been trained, run the clustering step. Provide the specific checkpoint name you want to use:

```
python run_clustering.py --config configs/supcon.yaml --ckpt_name last
```

The output of this step is a `.pkl` file that contains a python dict which maps speaker ids to cluster ids.

Having constructed the clusters, you can run any of the CHNS models (`chns.yaml`, `chns_hscl.yaml`) after updating the `data.init_args.batch_sampler_config.cluster_dict_path` argument in the config file:

```
python run_train.py --config configs/chns.yaml
```

## Test

To test any of the models on your desired test dataset (eg. Vox1-H) run:

```
python run_test.py --config configs/chns.yaml
```

Remember to specify the proper test data paths in the config file.

---

## Citation
If you find this work useful, please cite:

```bibtex
@article{masztalski2025_chns,
  title={Clustering-based hard negative sampling for supervised contrastive speaker verification},
  author={Masztalski, Piotr and Romaniuk, Michał and Żak, Jakub and Matuszewski, Mateusz and Kowalczyk, Konrad},
  journal={arXiv preprint arXiv:2507.17540},
  year={2025}
}
```

