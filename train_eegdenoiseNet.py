import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from Data_Preparation.data_prepare_eegdnet import prepare_data
from model_eegflownet import EEGFlowNet, MODEL_ARCH
from flow_matching import ConditionalFlowMatching, euler_solve
from metrics import CC, RRMSE, RRMSE_s, SNR as calc_SNR
from utils import train_cfm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEGFlowNet on EEGDenoiseNet")
    parser.add_argument("--config",  type=str, default="base_cfm.yaml",
                        help="YAML config filename inside ../config/")
    parser.add_argument("--device",  type=str, default="cuda:0")
    parser.add_argument("--n_type",  type=str, default="EOG",
                        choices=["EOG", "EMG", "ECG"])
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    n_steps_eval = config.get("cfm", {}).get("n_steps", 20)
    foldername   = os.path.join(os.path.dirname(__file__), "..",
                                "new_check_points", f"cfm_plus_{args.n_type}")
    os.makedirs(foldername, exist_ok=True)

   
    print("Preparing EEGDenoiseNet data ...")
    [X_train, y_train, X_test, y_test] = prepare_data(
        combin_num=11, train_per=0.9, noise_type=args.n_type, seed=args.seed
    )

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    train_val_set = TensorDataset(y_train, X_train)
    train_idx, val_idx = train_test_split(
        list(range(len(train_val_set))), test_size=0.2, random_state=args.seed
    )
    train_set = Subset(train_val_set, train_idx)
    val_set   = Subset(train_val_set, val_idx)

    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=config["train"]["batch_size"],
                              drop_last=False, num_workers=0)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  "
          f"Test SNR levels: {X_test.shape[0]}  Test segs/level: {X_test.shape[1]}")

    arch  = MODEL_ARCH[args.n_type]
    model = EEGFlowNet(**arch).to(args.device)
    cfm   = ConditionalFlowMatching()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGFlowNet ({args.n_type}) parameters: {n_params:,}")

    
    train_cfm(
        model=model,
        cfm=cfm,
        config=config["train"],
        train_loader=train_loader,
        device=args.device,
        valid_loader=val_loader,
        valid_epoch_interval=config["train"].get("valid_epoch_interval", 10),
        foldername=foldername,
        n_steps_val=n_steps_eval,
        clip_value=0.5,
    )

    
    best_ckpt = os.path.join(foldername, "model_cfm.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=args.device,
                                     weights_only=False))
    model.eval()
    print(f"\nBest model loaded from: {best_ckpt}")

    snr_dB_levels   = np.linspace(-5.0, 5.0, num=11)
    per_snr_results = []

    print(f"\n{'SNR (dB)':>10}  {'CC':>8}  {'RRMSE_t':>10}  {'RRMSE_f':>10}  {'SNR_out':>10}")
    print("-" * 55)

    for snr_idx in range(X_test.shape[0]):
        x_snr = torch.FloatTensor(X_test[snr_idx])
        y_snr = torch.FloatTensor(y_test[snr_idx])

        test_ds     = TensorDataset(y_snr, x_snr)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        all_clean, all_noisy, all_denoised = [], [], []
        with torch.no_grad():
            for clean_b, noisy_b in test_loader:
                noisy_b = noisy_b.to(args.device)
                x_out   = euler_solve(model, noisy_b, n_steps=n_steps_eval)
                all_clean.append(clean_b.numpy())
                all_noisy.append(noisy_b.cpu().numpy())
                all_denoised.append(x_out.cpu().numpy())

        all_clean    = np.concatenate(all_clean,    axis=0)
        all_noisy    = np.concatenate(all_noisy,    axis=0)
        all_denoised = np.concatenate(all_denoised, axis=0)

        cc_val  = CC(all_denoised, all_clean)
        rrmse_t = RRMSE(all_denoised, all_clean)
        rrmse_f = RRMSE_s(all_denoised, all_clean)
        snr_val = calc_SNR(all_clean.squeeze(), all_denoised.squeeze())

        per_snr_results.append({
            "snr_dB":  snr_dB_levels[snr_idx],
            "CC":      cc_val,
            "RRMSE_t": rrmse_t,
            "RRMSE_f": rrmse_f,
            "SNR":     snr_val,
        })

        print(f"{snr_dB_levels[snr_idx]:>+10.1f}  "
              f"{cc_val:>8.4f}  {rrmse_t:>10.4f}  "
              f"{rrmse_f:>10.4f}  {snr_val:>10.4f}")

        snr_folder = os.path.join(foldername, f"test_snr_{snr_idx}")
        os.makedirs(snr_folder, exist_ok=True)
        np.save(os.path.join(snr_folder, "test_clean.npy"),    all_clean)
        np.save(os.path.join(snr_folder, "test_noisy.npy"),    all_noisy)
        np.save(os.path.join(snr_folder, "test_denoised.npy"), all_denoised)

    print("-" * 55)
    for key in ["CC", "RRMSE_t", "RRMSE_f", "SNR"]:
        vals = [r[key] for r in per_snr_results]
        print(f"{'Mean ' + key:>22}:  {np.mean(vals):.4f}")

    csv_path = os.path.join(foldername, f"test_results_{args.n_type}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_snr_results[0].keys())
        writer.writeheader()
        writer.writerows(per_snr_results)
    print(f"\nResults saved to: {csv_path}")
    print(f"Checkpoints  in : {foldername}")
