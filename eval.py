import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from semilearn import get_net_builder, get_dataset, get_config, get_algorithm
import semilearn.nets as nets
from torchvision.models import resnet101 as torchvision_resnet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, required=True, help='Path to the config file (same as training)')
    parser.add_argument('--load_path', type=str, required=True, help='Path to the model_best.pth')

    # Parse arguments and load config
    _args = parser.parse_args()
    import yaml
    with open(_args.c, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 2. Pass the dictionary to get_config
    args = get_config(config_dict)

    # 3. (Optional but recommended) Manually override with CLI args if needed
    # This ensures your command line flags (like --gpu) take precedence
    for key, value in vars(_args).items():
        if value is not None:
            setattr(args, key, value)

    # Set default for amp if it's missing
    if not hasattr(args, 'amp'):
        args.amp = False

    # --- ADD THESE LINES ---
    if not hasattr(args, 'net_conf'):
        args.net_conf = None

    if not hasattr(args, 'num_labels'):
        args.num_labels = None
    # -----------------------

    # Override with load_path
    args.load_path = _args.load_path
    args.resume = True

    # Force GPU 0 if not set
    if not hasattr(args, 'gpu'):
        args.gpu = 0
    torch.cuda.set_device(args.gpu)

    print(f"Loading config from: {_args.c}")
    print(f"Loading model from: {args.load_path}")

    nets.resnet101 = resnet101_builder

    # 1. Build Model (Architecture only)
    net_builder = get_net_builder(args.net, args.net_conf)

    # Pass 'net_builder' (the function), NOT 'net' (the instance)
    algorithm = get_algorithm(args, net_builder, tb_log=None, logger=None)

    # 3. Load Checkpoint (Weights)
    checkpoint = torch.load(args.load_path, map_location='cpu')
    algorithm.load_model(args.load_path)
    algorithm.cuda()
    algorithm.eval()

    # 4. Load Dataset (Target Test Set)
    # Note: We specifically request the 'test' split which corresponds to your 'test' folder
    dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    test_dset = dataset_dict['test']
    eval_loader = DataLoader(test_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(test_dset)} samples...")

    # 5. Run Inference
    preds = []
    targets = []

    with torch.no_grad():
        for data in eval_loader:
            image = data['x_lb']
            target = data['y_lb']

            image = image.cuda()

            # Forward pass
            # Note: Semi-learn models output a dictionary or raw logits depending on mode
            logits = algorithm.model(image)

            # Get predictions
            prob = torch.softmax(logits, dim=-1)
            pred = prob.argmax(1)

            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())

    # 6. Calculate Metrics
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    acc = (preds == targets).mean() * 100
    f1 = f1_score(targets, preds, average='macro') * 100

    print("-" * 30)
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Test Macro-F1: {f1:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()