import csv
import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import argparse

from config import cfg, update_cfg
from dataset import get_dataloader
from world_model import HTPCWM
from visualize import plot_reconstruction, plot_rollout, plot_latent_heatmap, plot_training_curves


VIZ_INTERVAL = 1


def find_latest_checkpoint(save_dir):
    checkpoints = []
    for f in os.listdir(save_dir):
        if f.startswith("model_epoch_") and f.endswith(".pt"):
            epoch = int(f.replace("model_epoch_", "").replace(".pt", ""))
            checkpoints.append((epoch, os.path.join(save_dir, f)))
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1], checkpoints[-1][0]


def print_model_summary(model):
    print("=" * 60)
    print("Model: HT-PCWM (Hierarchical Temporal Predictive Coding World Model)")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print("-" * 60)
    print("Layer breakdown:")

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                print(f"  {name:40s} {layer_params:>12,}")

    print("=" * 60)


def run_fast_visualizations(model, val_loader, device, save_dir, epoch):
    print(f"  Generating visualizations for epoch {epoch}...")
    viz_dir = os.path.join(save_dir, "visualizations", f"epoch_{epoch}")
    os.makedirs(viz_dir, exist_ok=True)

    plot_reconstruction(model, val_loader, device, viz_dir)
    plot_rollout(model, val_loader, device, viz_dir)
    plot_latent_heatmap(model, val_loader, device, viz_dir)
    plot_training_curves(save_dir, viz_dir)

    print(f"  Saved to {viz_dir}")


class MetricsTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_path = os.path.join(save_dir, "metrics.csv")
        self.epochs = []
        self.train_losses = []
        self.train_frame_losses = []
        self.train_latent1_losses = []
        self.train_latent2_losses = []
        self.train_iterations = []
        self.val_losses = []
        self.val_frame_losses = []
        self.val_latent1_losses = []
        self.val_latent2_losses = []
        self.val_iterations = []
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 11:  # Updated for new columns
                        self.epochs.append(int(row[0]))
                        self.train_losses.append(float(row[1]))
                        self.train_frame_losses.append(float(row[2]))
                        self.train_latent1_losses.append(float(row[3]))
                        self.train_latent2_losses.append(float(row[4]))
                        self.train_iterations.append(float(row[5]))
                        self.val_losses.append(float(row[6]))
                        self.val_frame_losses.append(float(row[7]))
                        self.val_latent1_losses.append(float(row[8]))
                        self.val_latent2_losses.append(float(row[9]))
                        self.val_iterations.append(float(row[10]))

    def log_epoch(self, epoch, train_metrics, val_metrics):
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics["energy_total"])
        self.train_frame_losses.append(train_metrics["energy_frame"])
        self.train_latent1_losses.append(train_metrics["energy_latent1"])
        self.train_latent2_losses.append(train_metrics["energy_latent2"])
        self.train_iterations.append(train_metrics.get("avg_iterations", 5.0))
        self.val_losses.append(val_metrics["energy_total"])
        self.val_frame_losses.append(val_metrics["energy_frame"])
        self.val_latent1_losses.append(val_metrics["energy_latent1"])
        self.val_latent2_losses.append(val_metrics["energy_latent2"])
        self.val_iterations.append(val_metrics.get("avg_iterations", 5.0))
        self.save()

    def save(self):
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_energy_total", "train_energy_frame", "train_energy_latent1", "train_energy_latent2", "train_avg_iterations",
                "val_energy_total", "val_energy_frame", "val_energy_latent1", "val_energy_latent2", "val_avg_iterations"
            ])
            for i in range(len(self.epochs)):
                writer.writerow([
                    self.epochs[i],
                    self.train_losses[i], self.train_frame_losses[i],
                    self.train_latent1_losses[i], self.train_latent2_losses[i],
                    self.train_iterations[i],
                    self.val_losses[i], self.val_frame_losses[i],
                    self.val_latent1_losses[i], self.val_latent2_losses[i],
                    self.val_iterations[i]
                ])


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs to train")
    args = parser.parse_args()

    if args.epochs is not None:
        update_cfg(epochs=args.epochs)

    start_epoch = 0

    if args.resume or args.checkpoint:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
            start_epoch = int(checkpoint_path.split("_")[-1].replace(".pt", ""))
        else:
            checkpoint_path, start_epoch = find_latest_checkpoint(cfg.save_dir)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path} (epoch {start_epoch})")
            device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
            model = HTPCWM(cfg).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            start_epoch += 1
        else:
            print("No checkpoint found, starting from scratch")
            device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
            model = HTPCWM(cfg).to(device)
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            start_epoch = 0
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model = HTPCWM(cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print_model_summary(model)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)
    tracker = MetricsTracker(cfg.save_dir)

    print(f"Using device: {device}")
    print(f"Training from epoch {start_epoch} to epoch {cfg.epochs}")

    train_loader = get_dataloader(
        cfg.data_dir, cfg.train_split, cfg.batch_size, num_workers=0, sequence_length=cfg.sequence_length
    )
    val_loader = get_dataloader(
        cfg.data_dir, cfg.val_split, cfg.val_batch_size, num_workers=0, sequence_length=cfg.sequence_length
    )

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        train_total = 0.0
        train_frame = 0.0
        train_latent1 = 0.0
        train_latent2 = 0.0
        train_iterations = 0.0
        train_count = 0

        num_batches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=num_batches,
                   desc=f"Epoch [{epoch}/{cfg.epochs}]", leave=True)

        for batch_idx, video in pbar:
            video = video.to(device)
            batch_size, seq_len, channels, height, width = video.shape

            optimizer.zero_grad()
            total_loss = 0.0
            total_frame_loss = 0.0
            total_latent1_loss = 0.0
            total_latent2_loss = 0.0
            total_iterations = 0

            for t in range(seq_len - 1):
                frame_current = video[:, t]
                frame_next = video[:, t + 1]

                result = model(frame_current, frame_next)

                total_loss += result["energy_total"]
                total_frame_loss += result["energy_frame"]
                total_latent1_loss += result["energy_latent1"]
                total_latent2_loss += result["energy_latent2"]
                total_iterations += result.get("iterations_used", cfg.adaptive_max_steps)

            avg_loss = total_loss / (seq_len - 1)
            avg_frame_loss = total_frame_loss / (seq_len - 1)
            avg_latent1_loss = total_latent1_loss / (seq_len - 1)
            avg_latent2_loss = total_latent2_loss / (seq_len - 1)
            avg_iterations = total_iterations / (seq_len - 1)

            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += avg_loss.item()
            train_frame += avg_frame_loss.item()
            train_latent1 += avg_latent1_loss.item()
            train_latent2 += avg_latent2_loss.item()
            train_iterations += avg_iterations
            train_count += 1

            pbar.set_postfix({"loss": f"{avg_loss.item():.4f}",
                             "frame": f"{avg_frame_loss.item():.4f}",
                             "iter": f"{avg_iterations:.1f}"})

        pbar.close()
        train_metrics = {
            "energy_total": train_total / train_count,
            "energy_frame": train_frame / train_count,
            "energy_latent1": train_latent1 / train_count,
            "energy_latent2": train_latent2 / train_count,
            "avg_iterations": train_iterations / train_count,
        }

        writer.add_scalar("train/energy_total", train_metrics["energy_total"], epoch)
        writer.add_scalar("train/energy_frame", train_metrics["energy_frame"], epoch)
        writer.add_scalar("train/energy_latent1", train_metrics["energy_latent1"], epoch)
        writer.add_scalar("train/energy_latent2", train_metrics["energy_latent2"], epoch)
        writer.add_scalar("train/avg_iterations", train_metrics["avg_iterations"], epoch)

        model.eval()
        val_total = 0.0
        val_frame = 0.0
        val_latent1 = 0.0
        val_latent2 = 0.0
        val_iterations = 0.0
        val_count = 0

        with torch.no_grad():
            for video in val_loader:
                video = video.to(device)
                batch_size, seq_len, channels, height, width = video.shape
                total_loss = 0.0
                total_frame_loss = 0.0
                total_latent1_loss = 0.0
                total_latent2_loss = 0.0
                total_iterations = 0

                for t in range(seq_len - 1):
                    frame_current = video[:, t]
                    frame_next = video[:, t + 1]
                    result = model(frame_current, frame_next)
                    total_loss += result["energy_total"]
                    total_frame_loss += result["energy_frame"]
                    total_latent1_loss += result["energy_latent1"]
                    total_latent2_loss += result["energy_latent2"]
                    total_iterations += result.get("iterations_used", cfg.adaptive_max_steps)

                val_total += (total_loss / (seq_len - 1)).item()
                val_frame += (total_frame_loss / (seq_len - 1)).item()
                val_latent1 += (total_latent1_loss / (seq_len - 1)).item()
                val_latent2 += (total_latent2_loss / (seq_len - 1)).item()
                val_iterations += total_iterations / (seq_len - 1)
                val_count += 1

        val_metrics = {
            "energy_total": val_total / val_count,
            "energy_frame": val_frame / val_count,
            "energy_latent1": val_latent1 / val_count,
            "energy_latent2": val_latent2 / val_count,
            "avg_iterations": val_iterations / val_count,
        }

        writer.add_scalar("val/energy_total", val_metrics["energy_total"], epoch)
        writer.add_scalar("val/energy_frame", val_metrics["energy_frame"], epoch)
        writer.add_scalar("val/energy_latent1", val_metrics["energy_latent1"], epoch)
        writer.add_scalar("val/energy_latent2", val_metrics["energy_latent2"], epoch)
        writer.add_scalar("val/avg_iterations", val_metrics["avg_iterations"], epoch)

        tracker.log_epoch(epoch, train_metrics, val_metrics)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch}/{cfg.epochs}] "
              f"Train: {train_metrics['energy_total']:.4f} | "
              f"Val: {val_metrics['energy_total']:.4f} | "
              f"Iter: {train_metrics['avg_iterations']:.1f} | "
              f"Time: {epoch_time:.1f}s")

        if True:
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"model_epoch_{epoch+1}.pt"))
            run_fast_visualizations(model, val_loader, device, cfg.save_dir, epoch + 1)

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "model_final.pt"))
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
