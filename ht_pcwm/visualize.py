import csv
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import make_grid, save_image

from config import cfg
from dataset import get_dataloader, MovingMNISTVideo
from world_model import HTPCWM


def plot_reconstruction(model, val_loader, device, save_dir, num_frames=5):
    print("Generating reconstruction visualization...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for video in val_loader:
            if isinstance(video, tuple):
                video = video[0]
            video = video.to(device)
            break

    frames_to_show = min(num_frames, video.shape[1] - 1)
    rows = frames_to_show
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(frames_to_show):
            frame_current = video[0, i:i+1]
            frame_next = video[0, i+1:i+2]
            result = model(frame_current, frame_next)

            frame_pred = result["frame_prediction"]
            frame_current_np = ((frame_current[0, 0].cpu() + 1) / 2).clamp(0, 1)
            frame_pred_np = ((frame_pred[0, 0].cpu() + 1) / 2).clamp(0, 1)
            frame_next_np = ((frame_next[0, 0].cpu() + 1) / 2).clamp(0, 1)

            axes[i, 0].imshow(frame_current_np, cmap='gray')
            axes[i, 0].set_title("Current Frame" if i == 0 else "")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(frame_pred_np, cmap='gray')
            axes[i, 1].set_title("Prediction" if i == 0 else "")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(frame_next_np, cmap='gray')
            axes[i, 2].set_title("Target" if i == 0 else "")
            axes[i, 2].axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, "reconstruction_grid.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved reconstruction to {path}")


def plot_rollout(model, val_loader, device, save_dir, num_steps=10):
    print("Generating rollout visualization...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for video in val_loader:
            if isinstance(video, tuple):
                video = video[0]
            video = video.to(device)
            break

    frame_start = video[0, 0:1]
    actual_frames = video[0, 1:num_steps+1]

    with torch.no_grad():
        predictions = model.rollout(frame_start, num_steps=num_steps)
        predictions = predictions.squeeze(0)

    max_show = min(num_steps, 5)
    total_cols = max_show + 1
    fig, axes = plt.subplots(2, total_cols, figsize=(total_cols * 3, 6))

    axes[0, 0].imshow(((frame_start[0, 0].cpu() + 1) / 2).clamp(0, 1), cmap='gray')
    axes[0, 0].set_title("Start Frame")
    axes[0, 0].axis('off')

    for i in range(max_show):
        axes[0, i+1].imshow(((predictions[i].cpu().squeeze(0) + 1) / 2).clamp(0, 1), cmap='gray')
        axes[0, i+1].set_title(f"Pred +{i+1}")
        axes[0, i+1].axis('off')

    axes[1, 0].axis('off')
    for i in range(max_show):
        axes[1, i+1].imshow(((actual_frames[i].cpu().squeeze(0) + 1) / 2).clamp(0, 1), cmap='gray')
        axes[1, i+1].set_title(f"Actual +{i+1}")
        axes[1, i+1].axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, "rollout_sequence.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved rollout to {path}")


def plot_pca_latent_space(model, val_loader, device, save_dir, num_samples=2000):
    print("Generating PCA latent space visualizations...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    print(f"Extracting latent2 features from {num_samples} samples...")
    features = []
    motion_types = []
    primary_directions = []
    primary_digits = []
    num_digits_list = []

    dataset = MovingMNISTVideo(cfg.data_dir, cfg.val_split, cfg.sequence_length, load_metadata=True)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in indices:
            video, metadata = dataset[idx]
            frame = video[0].unsqueeze(0).to(device)

            latent2 = model.hierarchy.downsample_latent1(
                model.hierarchy.encode(frame)
            )

            feat = latent2.flatten().cpu().numpy()
            features.append(feat)

            motion_types.append(metadata['motion_type'])
            dirs = metadata['directions']
            primary_directions.append(dirs[0] if dirs else 'unknown')
            dl = metadata['digit_labels']
            primary_digits.append(dl[0] if dl else 0)
            num_digits_list.append(len(dl))

            if len(features) % 200 == 0:
                print(f"  Processed {len(features)}/{num_samples}")

    features = np.array(features)
    print(f"Feature matrix shape: {features.shape}")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    pca_data = {
        'pca_motion_type.png': {
            'title': 'PCA - Motion Type',
            'labels': motion_types,
            'unique': sorted(set(motion_types)),
            'cmap': 'tab10'
        },
        'pca_directions.png': {
            'title': 'PCA - Direction',
            'labels': primary_directions,
            'unique': sorted(set(primary_directions)),
            'cmap': 'tab10'
        },
        'pca_digit_labels.png': {
            'title': 'PCA - Digit Label',
            'labels': primary_digits,
            'unique': sorted(set(primary_digits)),
            'cmap': 'tab10'
        },
        'pca_num_digits.png': {
            'title': 'PCA - Number of Digits',
            'labels': num_digits_list,
            'unique': sorted(set(num_digits_list)),
            'cmap': 'tab10'
        },
    }

    for filename, meta in pca_data.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = meta['unique']
        n_colors = len(unique_labels)
        cmap = plt.cm.get_cmap(meta['cmap'], n_colors)

        for i, label in enumerate(unique_labels):
            mask = [l == label for l in meta['labels']]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[cmap(i)], label=str(label), alpha=0.6, s=10)

        ax.set_title(meta['title'])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {filename}")

    print("PCA visualizations complete!")


def plot_training_curves(log_dir, save_dir):
    print("Generating training curves...")
    os.makedirs(save_dir, exist_ok=True)

    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics.csv found at {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data in metrics.csv")
        return

    epochs = [int(r['epoch']) for r in rows]
    train_total = [float(r['train_energy_total']) for r in rows]
    train_frame = [float(r['train_energy_frame']) for r in rows]
    train_latent1 = [float(r['train_energy_latent1']) for r in rows]
    train_latent2 = [float(r['train_energy_latent2']) for r in rows]
    val_total = [float(r['val_energy_total']) for r in rows]
    val_frame = [float(r['val_energy_frame']) for r in rows]
    val_latent1 = [float(r['val_energy_latent1']) for r in rows]
    val_latent2 = [float(r['val_energy_latent2']) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_total, label='Train')
    axes[0, 0].plot(epochs, val_total, label='Val')
    axes[0, 0].set_title('Energy Total')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_frame, label='Train')
    axes[0, 1].plot(epochs, val_frame, label='Val')
    axes[0, 1].set_title('Energy Frame')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, train_latent1, label='Train')
    axes[1, 0].plot(epochs, val_latent1, label='Val')
    axes[1, 0].set_title('Energy Latent1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, train_latent2, label='Train')
    axes[1, 1].plot(epochs, val_latent2, label='Val')
    axes[1, 1].set_title('Energy Latent2')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Curves', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves to {path}")


def plot_latent_heatmap(model, val_loader, device, save_dir, num_frames=20, top_channels=16):
    print("Generating latent heatmap...")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for video in val_loader:
            if isinstance(video, tuple):
                video = video[0]
            video = video.to(device)
            break

    latent_activations = []

    with torch.no_grad():
        for t in range(min(num_frames, video.shape[1] - 1)):
            frame_current = video[:, t]
            frame_next = video[:, t + 1]
            result = model(frame_current, frame_next)

            latent1 = result["latent1"]
            pooled = torch.mean(latent1, dim=[2, 3])
            latent_activations.append(pooled[0].cpu().numpy())

    latent_activations = np.array(latent_activations)
    print(f"Latent activations shape: {latent_activations.shape}")

    mean_activation = latent_activations.mean(axis=0)
    top_channel_indices = np.argsort(mean_activation)[-top_channels:][::-1]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, ch_idx in enumerate(top_channel_indices):
        ax = axes[i]
        heatmap = latent_activations[:, ch_idx]
        ax.plot(heatmap, linewidth=2)
        ax.set_title(f"Channel {ch_idx}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Activation")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Top {top_channels} Most Active latent1 Channels Over Time", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "latent1_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved latent heatmap to {path}")


def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    viz_dir = os.path.join(cfg.save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    model = HTPCWM(cfg).to(device)
    checkpoint_path = os.path.join(cfg.save_dir, "model_final.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("WARNING: No checkpoint found, using randomly initialized model!")

    val_loader = get_dataloader(
        cfg.data_dir, cfg.val_split,
        batch_size=cfg.val_batch_size,
        num_workers=2,
        sequence_length=cfg.sequence_length
    )

    plot_reconstruction(model, val_loader, device, viz_dir, num_frames=cfg.viz_reconstruction_frames)
    plot_rollout(model, val_loader, device, viz_dir, num_steps=cfg.viz_rollout_steps)
    plot_pca_latent_space(model, val_loader, device, viz_dir, num_samples=cfg.viz_pca_samples)
    plot_training_curves(cfg.save_dir, viz_dir)
    plot_latent_heatmap(model, val_loader, device, viz_dir,
                        num_frames=cfg.viz_heatmap_frames,
                        top_channels=cfg.viz_heatmap_channels)

    print(f"\nAll visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
