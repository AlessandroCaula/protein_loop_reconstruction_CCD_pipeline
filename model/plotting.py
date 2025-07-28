import matplotlib.pyplot as plt
import numpy as np

def plot_backbone(backbone, title="Backbone"):
    CA_coords = np.array([res["CA"] for res in backbone])
    xs, ys, zs = CA_coords[:, 0], CA_coords[:, 1], CA_coords[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker='o')
    ax.set_title(title)
    plt.show()

def plot_anchor_and_backbone(anchor, target, backbone, title="Anchors and Backbone"):
    CA_coords_backbone = np.array([res["CA"] for res in backbone])
    xs, ys, zs = CA_coords_backbone[:, 0], CA_coords_backbone[:, 1], CA_coords_backbone[:, 2]

    fig = plt.figure()
    ax= fig.add_subplot(111, projection='3d')
    # Plot backbone CA trace
    ax.plot(xs, ys, zs, marker='o', color='blue', label='Backbone CA')
    # Plot anchor point
    ax.scatter(anchor[0], anchor[1], anchor[2], color='red', label='Anchor')
    # Plot target point
    ax.scatter(target[0], target[1], target[2], color='green', label='Target')
    ax.set_title(title)
    ax.legend()
    plt.show()