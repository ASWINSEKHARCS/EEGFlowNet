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

from Data_Preparation.data_prepare_ssed import prepare_data_ssed
from model_eegflownet import EEGFlowNet, MODEL_ARCH
from flow_matching import ConditionalFlowMatching, euler_solve
from metrics import CC, RRMSE, RRMSE_s, SNR as calc_SNR
from utils import train_cfm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEGFlowNet on SSED dataset")
    parser.add_argument("--config",  type=str, default="ssed_cfm.yaml",
                        help="YAML config filename inside ../config/")
    parser.add_argument("--device",  type=str, default="cuda:0")
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
                                "new_check_points", "cfm_ssed")
    os.makedirs(foldername, exist_ok=True)

    print("Preparing SSED data ...")
    [X_train_all, y_train_all, X_test, y_test] = prepare_data_ssed(
        train_per=0.9, seed=args.seed
    )

    X_train_all = torch.FloatTensor(X_train_all)   
    y_train_all = torch.FloatTensor(y_train_all)   

    train_val_set = TensorDataset(y_train_all, X_train_all)
    train_idx, val_idx = train_test_split(
        list(range(len(train_val_set))), test_size=0.2, random_state=args.seed
    )
    train_set = Subset(train_val_set, train_idx)
    val_set   = Subset(train_val_set, val_idx)

    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=config["train"]["batch_size"],
                              drop_last=False, num_workers=0)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {X_test.shape[0]}")

    arch  = MODEL_ARCH["EOG"]  
    model = EEGFlowNet(**arch).to(args.device)
    cfm   = ConditionalFlowMatching()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGFlowNet (SSED) parameters: {n_params:,}")

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
        clip_value=1.0,
        save_on="CC",
    )

    x_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    test_ds     = TensorDataset(y_test_t, x_test_t)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    ckpts_to_eval = [
        ("model_cfm.pth",  "Best-CC checkpoint"),
        ("final_cfm.pth",  "Final-epoch checkpoint"),
    ]

    all_rows = []
    for ckpt_name, ckpt_label in ckpts_to_eval:
        ckpt_path = os.path.join(foldername, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  {ckpt_label}: not found, skipping")
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=args.device,
                                          weights_only=False))
        model.eval()
        print(f"\nLoaded: {ckpt_path}")

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

        print(f"  [{ckpt_label}]")
        print(f"    CC      : {cc_val:.4f}")
        print(f"    RRMSE_t : {rrmse_t:.4f}")
        print(f"    RRMSE_f : {rrmse_f:.4f}")
        print(f"    SNR     : {snr_val:.4f} dB")

        tag = ckpt_name.replace(".pth", "")
        np.save(os.path.join(foldername, f"test_clean_{tag}.npy"),    all_clean)
        np.save(os.path.join(foldername, f"test_noisy_{tag}.npy"),    all_noisy)
        np.save(os.path.join(foldername, f"test_denoised_{tag}.npy"), all_denoised)
        all_rows.append({"checkpoint": ckpt_label, "CC": cc_val,
                         "RRMSE_t": rrmse_t, "RRMSE_f": rrmse_f, "SNR_dB": snr_val})

    csv_path = os.path.join(foldername, "test_results_ssed.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "CC", "RRMSE_t", "RRMSE_f", "SNR_dB"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to : {csv_path}")
    print(f"Checkpoints in   : {foldername}")
