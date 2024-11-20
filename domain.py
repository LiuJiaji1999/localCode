import numpy as np
import ot
from torch import cdist
import torch
import ot

def compute_swd(features_1, features_2):
    """
    Compute Sliced Wasserstein Distance (SWD) between two feature sets.
    Args:
        features_1 (torch.Tensor): Tensor of shape (n_samples, n_projections) for dataset 1.
        features_2 (torch.Tensor): Tensor of shape (n_samples, n_projections) for dataset 2.
    Returns:
        float: The Sliced Wasserstein Distance.
    """
    n_proj = features_1.shape[1]

    # Precompute uniform distributions for OT
    unif_1 = torch.ones(features_1.shape[0], dtype=torch.float32) / features_1.shape[0]
    unif_2 = torch.ones(features_2.shape[0], dtype=torch.float32) / features_2.shape[0]

    # Reshape features for batch processing
    feas_1 = features_1.unsqueeze(-1)  # (n_samples, n_proj, 1)
    feas_2 = features_2.unsqueeze(-1)  # (n_samples, n_proj, 1)

    # Batch OT computation
    wasserstein_distances = 0
    for i in range(n_proj):
        # Select the i-th projection for both feature sets
        proj_feas_1 = feas_1[:, i, :]
        proj_feas_2 = feas_2[:, i, :]

        # Compute the pairwise cost matrix
        cost_matrix = torch.cdist(proj_feas_1, proj_feas_2, p=2)  # Squared L2 distance

        # Convert to NumPy for OT computation (if necessary)
        cost_matrix_np = cost_matrix.cpu().numpy()
        unif_1_np = unif_1.cpu().numpy()
        unif_2_np = unif_2.cpu().numpy()

        # Compute the OT plan using POT
        gamma = ot.emd(unif_1_np, unif_2_np, cost_matrix_np, numItermax=1e6)

        # Compute Wasserstein distance for the current projection
        wasserstein_distances += np.sum(gamma * cost_matrix_np)

    # Compute the mean Wasserstein distance across all projections
    # return np.mean(wasserstein_distances)
    return wasserstein_distances /  n_proj


# 示例代码：假设 features_1 和 features_2 是从模型中提取的特征
features_1 = torch.rand(100, 128)  # 100 samples, 128 projections
features_2 = torch.rand(100, 128)

swd = compute_swd(features_1, features_2)
print(f"Sliced Wasserstein Distance: {swd}")


# # 判断矩阵中每个值是否为 0
# is_zero_matrix = (gamma == 0)
# print("Matrix of True (if value is 0) and False (otherwise):")
# print(is_zero_matrix)

# # 判断整个矩阵是否全为 0
# if np.all(gamma == 0):
#     print("The entire gamma matrix is zero.")
# else:
#     print("The gamma matrix contains non-zero values.")

# # # 判断矩阵是否包含任意非零值
# # if np.any(gamma != 0):
# #     print("The gamma matrix contains non-zero values.")
# # else:
# #     print("The gamma matrix is entirely zero.")