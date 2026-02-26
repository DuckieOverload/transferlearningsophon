# transferlearningsophon

## Transfer Learning Project

This README page aims to be an introduction for the ongoing transfer learning project with applications to particle physics. The project in it of itself is aimed at advancing and observing the machine learning applications to the world of particle physics, and specifically, to the task of jet-tagging. As described in the original repository for Sophon, _"...the model Sophon (Sophon (Signature-Oriented Pre-training for Heavy-resonance ObservatioN) is a method proposed for developing foundation AI models tailored for future usage in LHC experimental analyses..." _ More specifically, the Sophon is a deep learning framework developed with the goal of better classifiying jets—AKA, collimated sprays of particles produced in high-energy collisions at places like the LHC (Large Hadron Collider)—using both particle-level and jet-level features.

The bigger and more universal goal, however, is to explore representation learning in jet physics, focusing on how neuralnetowrk embeddings capture physical information across different datasets and simulation domains. By doing all of this, we are aiming for the following, overarching goal: Evaluate transfer learning potential across deep learning models and jet types (Sophon vs. ParT & Higgs, top, QCD, etc.)

This README file focuses on one core task: running inference with Sophon on a subset of the JetClass dataset (which can be accessed through https://zenodo.org/records/6619768 -> "JetClass_Pythia_val_5M.tar" -> Download & Extract) and extracting embeddings for visualization and classification. It is written to simplify and better explain the whole process.

### Steps to follow:
1. Set up a new Python venv for the project
2. Download and unzip the data from the .tar file and save to an accessible folder for the project (https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1)
3. Create a new .py file in the /sophon folder to run the model (model located in: example_ParticleTransforme_sophon.py file).
4. Run the inference script (reads ROOT files + writes a CSV file with the embeddings)
5. Explore embeddings (simple plotting)

## Requirements
- Python 3.10+
- PyTorch
- uproot, numpy, tqdm, awkward

## Install for the new venv:
```sh
from repo_root/
conda create -n sophon python=3.10 -y
conda activate sophon
# Install PyTorch (pick the right command for your CUDA)
# See https://pytorch.org/get-started/locally/ for your platform; example (CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Core deps
pip install uproot numpy tqdm
```

## Data
Once you have downlaoded the subset of the JetClass dataset, place the .root files in data/JetClass/val_5M. The example config file as well as the inference scrip, expect around 5 of the validation files to successfully run inference on them: 
```
HToBB_120.root, HToBB_121.root,...,HToBB_124.root
```

## Inference script

Access the inference_jetclass.py file in this repository and download. The inference script:
1. Reads particle-flow and scalar features from each event in the .root files
2. Pads up to the 128 mparticles and skips events/logits events
3. Calls the function getm_model from example_ParticleTransformer_sophon.py
4. Writes CSV files with: file name, event index, truth label, some of the main kinematic features, and a 128-D vector embedding
5. From a local terminal, run:
```sh
python inference_with_embedding.py
```
_It will take a couple minutes per root file_

### Steps to follow in order to successfully run data through Sophon

The steps that are required to be followed in order to run a comparison between QCD (in this case, background noise) and any of the other jet classes available in the val_5M dataset through Sophon are as follows:
1. Select which jet class to compare to QCD.
- For example, let's take the HToCC files.
One will notice that in the val_5M dataset, there are a total of five files related to the HToCC class; all labeled as "HToCC_120.root", "HToCC_121.root",...,"HToCC_124.root". When feeding these to Sophon — or any other jet class, really — we must include all five files in our code as such:
```sh
root_files = ["HToCC_120.root", "HToCC_121.root","HToCC_122.root","HToCC_123.root","HToCC_124.root"]
```
2. Make sure you pf_keys is the correct dimension.
- If the model receives any other dimensions when it comes to what we are feeding it, it will not run.
Make sure pf_keys is composed of particle_keys + scalar_keys in order to run the data through the model successfully. Consequently, particle_keys and scalar_keys must each contain all labels that are in the inference script above. They must look like this:
```sh
particle_keys = [
        'part_px', 'part_py', 'part_pz', 'part_energy',
        'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
        'part_dzval', 'part_dzerr', 'part_charge',
        'part_isChargedHadron', 'part_isNeutralHadron',
        'part_isPhoton', 'part_isElectron', 'part_isMuon'
]
scalar_keys = [
        'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
        'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
        'label_Tbqq', 'label_Tbl', 'jet_pt', 'jet_eta', 'jet_phi',
        'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1',
        'jet_tau2', 'jet_tau3', 'jet_tau4', 'aux_genpart_eta',
        'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt',
        'aux_truth_match'
]
```
3. If you want a record of the inference, make sure to save it to a csv file.
- In order to successfully save the data being run through Sophon, the cleanest and most accessible way to do it is to save the inference output into a csv file under an appropriate name (e.g.: inference_sophon.csv)
To do this, we must first "import csv" at the very top of our inference script.
-Then, before running the inference loop we create a new variable under an appropriate name as such:
```sh
OUTPUT_CSV = "inference_sophon.csv"
```
-For this to actually create and write into the CSV file, we must open the file using python's csv.writer before the event loop, and define the header row we want. For example:
```sh
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # write header (following format is what we have been using and what works best so far)
        header = ["file", "event_index", "truth_label", "label_name"] + \
                         [f"emb_{j}" for j in range(128)]
        writer.writerow(header)

        # now begin your inference loop 
        for i in range(total_events):
                # compute embedding, truth label, etc.
                row = [file_name, i, truth_label, label_name] + list(embedding)
                writer.writerow(row)
```
In summary, the main points are:
1. Open the file using with open(...) before the loop.
2. Create the writer object.
3. Write in the header.
4. Write exactly one CSV row per event inside the inference loop.

### Command-line usage and parameters for `inference_jetclass.py`

The shipped script supports a small set of command-line options. Usage:

```
python inference_jetclass.py [OPTIONS] data_file1.root data_file2.root ...
```

Options:
- `-o`, `--output`: Path to output CSV file. Default: `HToCC_inference_with_embedding.csv`.
- `--root-dir`: Directory to prepend to data file names when a provided file name is not an existing path. Default: `../data/JetClass/val_5M` (the example dataset folder in this repo).
- `data_files` (positional): One or more ROOT file names or full paths to process. If a name is not found as-is, the script will join it with `--root-dir`.

Behaviour and important notes:
- Device selection: the script will use CUDA if available otherwise CPU.
- Model call: `get_model(..., export_embed=True)` is used to return the model (and optionally embedding). The script supports model outputs that are either `(logits, embedding)` or `embedding` only.
- Maximum particles: The script pads/truncates to `max_part = 128`. Events with more than 128 particles are skipped.

CSV output format:
- Header columns written by the script (in order):
    - `file`, `event_index`, `truth_label`, `label_name`, `jet_sdmass`, `jet_mass`, `jet_pt`, `jet_eta`, `jet_phi`
    - `prob_0` ... `prob_9` (10 model output probabilities when logits are returned)
    - `emb_0` ... `emb_127` (128-dimensional embedding vector)

Example invocations:

Process files by name (use `--root-dir` fallback):
```
python inference_jetclass.py -o HToCC_inference_with_embedding.csv HToCC_120.root HToCC_121.root HToCC_122.root
```

Process files by absolute path:
```
python inference_jetclass.py -o /path/to/out.csv /full/path/to/HToCC_120.root /full/path/to/HToCC_121.root
```

If you want to change behavior (e.g., `max_part` or model config) edit the script's constants near the top (`max_part = 128`) or modify the `DummyDataConfig` passed to `get_model` to match the model's expected inputs.

### Notebook: `mlp.ipynb` — parameters and how to adapt

The notebook trains an MLP on the 128-d embeddings produced by the inference script. Important variables and where to change them:

- Input CSV files: the notebook reads multiple CSV files (one per class) near the top in the data-loading cell. Edit these lines to point to your CSVs:
    - `df_qcd = pd.read_csv("ZJetsToNuNu_inference_with_embedding.csv")`
    - `df_hbb = pd.read_csv("HToBB_inference_with_embedding.csv")`
    - ... (one `pd.read_csv` per class)

- Alternate approach: The notebook includes a `class_files` mapping (commented) where you can map class ids to filenames and let the notebook build `df_all` for you. Uncomment and edit that mapping to change dataset sources.

- Splitting and sampling:
    - Stratified splits are used to create train/val/test sets. The notebook uses a 70/15/15 split by default (test_size=0.3 then split 50/50 of the temp set). To change proportions, edit the `train_test_split` / `StratifiedShuffleSplit` calls in the data prep cells.

- Model / training hyperparameters (cells labeled `Step 2/3/4`):
    - MLP architecture: the `MLP` class defines layer sizes (default 128→256→128→64→10). Edit the class to change hidden sizes.
    - Device: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` — will auto-detect GPU.
    - Loss & optimizer: `criterion = nn.CrossEntropyLoss()` and `optimizer = optim.Adam(model.parameters(), lr=1e-3)` — change learning rate or optimizer here.
    - Batch size: set in `DataLoader(..., batch_size=512)` — edit as needed.
    - Epochs: the training call `train_model(model, train_loader, val_loader, epochs=1)` sets `epochs`; increase for longer training.

- Evaluation: the notebook shows how to compute per-class ROC AUC using predictions on `X_test`. To change how probabilities are constructed, the notebook uses `torch.softmax(logits, dim=1)`; modify if using other score forms.

Examples (quick edits):

- To change batch size to 256, edit the loader creation:
```
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
```

- To train for 10 epochs with lr=5e-4:
```
optimizer = optim.Adam(model.parameters(), lr=5e-4)
train_model(model, train_loader, val_loader, epochs=10)
```

### Troubleshooting and tips
- If `uproot` fails to open a file, verify the path or use the full path to the ROOT file in the positional `data_files` list.
- If embeddings appear `NaN` or extremely large, ensure scalar and particle keys align with the model's expected input order (see `particle_keys` and `scalar_keys` in `inference_jetclass.py`).
- If you only want embeddings (no probs), ensure the model returns embedding-only or adapt the CSV header expectation accordingly.

If you'd like, I can also add a small example script that batches a list of files and runs the inference in parallel, or add a `requirements.txt` with the exact packages used by the notebook and script. Want me to add that?

### What `mlp.ipynb` does

High-level summary:
- Purpose: train a small Multi-Layer Perceptron (MLP) on the 128-dimensional embeddings produced by Sophon to classify the 10 JetClass categories and evaluate per-class discrimination.
- Inputs: one or more CSV files produced by `inference_jetclass.py` containing `truth_label`, `prob_*` (optional), and `emb_0`..`emb_127` columns.
- Outputs: a trained MLP classifier, training/validation logs, test accuracy, and per-class ROC/AUC plots.

Notebook workflow (what each major section does):
1. Data loading: reads several per-class CSV files into pandas, concatenates them, and extracts `X` (embedding columns `emb_0..emb_127`) and `y` (`truth_label`).
2. Train/val/test split: uses stratified splitting to preserve class balance (default: 70% train, 15% val, 15% test). The split code is editable in the first data cell.
3. Model definition: defines `MLP` with layers 128→256→128→64→10 (ReLU activations) and returns raw logits.
4. Training loop: trains with `CrossEntropyLoss` and `Adam` (default lr=1e-3), reports epoch loss and accuracy for train and validation sets. Batch size and epochs are set in the DataLoader and training call.
5. Evaluation: computes test accuracy and softmax probabilities on the test set; computes per-class ROC curves and AUC values using `sklearn.metrics`.
6. Visualization: plots per-class ROC curves (log axes options are included) and saves or displays figures.

How to adapt or re-run:
- Point the top data-loading cell to your CSV filenames (one `pd.read_csv` per class) or edit the `class_files` mapping and let the notebook build `df_all` programmatically.
- Change `batch_size`, `lr`, and `epochs` in the training cells to tune training.
- To reproduce results deterministically, set random seeds before splitting (e.g., `np.random.seed(42)` and torch seeds) and re-run the data-prep cells.

Expected runtime and resources:
- Small MLP on the 128-d embeddings is fast; on CPU a single epoch over tens of thousands of rows may take minutes, on GPU it's faster. The notebook auto-selects CUDA if available.

Notes:
- The notebook assumes the CSVs contain one embedding row per event. If your CSV contains extra columns, the notebook selects only `emb_0..emb_127` and `truth_label` for training.
- If you want a saved trained model, add `torch.save(model.state_dict(), "mlp_model.pt")` after training.
