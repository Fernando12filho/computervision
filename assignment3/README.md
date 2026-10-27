# CNN for MNIST Digit Classification

CAP 5410 (Computer Vision), Assignment 3. A LeNet-5 style CNN written in
PyTorch is trained on MNIST. The program records batch-level metrics for
the first epoch and epoch-level metrics for 10 epochs, compares three
learning rates, picks one using the validation set, and tests it once on
the MNIST test set.

## Data split

- 60,000 MNIST training images split into 48,000 training / 12,000
  validation, using a random permutation seeded with 42.
- The 10,000 test images are used once, at the end, on the selected model.
- Preprocessing: `ToTensor` then `Normalize(0.1307, 0.3082)`. The mean and
  standard deviation come from the 48,000 training images only.

## Requirements

- Python 3.13
- Dependencies (CPU build of PyTorch) are pinned in `requirements.txt`:

  ```
  python -m venv env
  env\Scripts\activate            # Windows
  source env/bin/activate         # Linux / macOS
  pip install -r requirements.txt
  ```

## Running it

```
python main.py
```

On the first run MNIST is downloaded to `./data`. The script then:

1. prints the environment and the per-layer parameter table;
2. trains three configurations for 10 epochs each (SGD, momentum 0.9,
   batch size 64, learning rates 0.01 (baseline), 0.001 and 0.1);
3. selects the configuration with the best validation accuracy after
   epoch 10;
4. evaluates that model once on the test set (accuracy, macro-precision,
   macro-recall, macro-F1);
5. saves every figure to `report/figures/`.

It takes about 10 minutes on an Intel Core i7-13700F (CPU only).
`python main.py --skip-train` reuses the saved results and only repeats
steps 3-5.

## Reproducibility

Seed 42 controls the split, the initial weights and the shuffle order.
Training runs on the CPU with `torch.use_deterministic_algorithms(True)`
and `num_workers=0`, so a second run with the same library versions gives
the same numbers.

## Files

| File | Contents |
|---|---|
| `dataloader.py` | MNIST download, 48k/12k split, normalization, DataLoaders |
| `model.py` | The CNN (`Net`) and the parameter-count helpers |
| `train.py` | Training loop, evaluation, macro metrics, test evaluation |
| `plots.py` | All report figures |
| `main.py` | Runs everything in order |
| `requirements.txt` | Pinned dependency versions |
| `results/` | Created on run: output log, per-configuration histories and weights, test metrics |
