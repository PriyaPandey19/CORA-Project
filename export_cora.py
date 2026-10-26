import numpy as np
from torch_geometric.datasets import Planetoid
d = Planetoid(root="data", name="Cora")[0]
np.savez_compressed("cora_data.npz", x=d.x.numpy().astype("float32"), edge_index=d.edge_index.numpy().astype("int64"))
print(d.x.shape, d.edge_index.shape)