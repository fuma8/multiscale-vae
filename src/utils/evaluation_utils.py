import os

import matplotlib.pyplot as plt

def visualize_images_grid(images, save_path, num_images=16, grid_size=(4, 4)):

    _, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1]*2, grid_size[0]*2))

    for i in range(num_images):
        img = images[i].detach().cpu().permute(1, 2, 0).clamp(0, 1)  # (3,H,W) → (H,W,3)
        ax = axs[i // grid_size[1], i % grid_size[1]]
        ax.imshow(img.numpy())
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved results: {os.path.abspath(save_path)}")
