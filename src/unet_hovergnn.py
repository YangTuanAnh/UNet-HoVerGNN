import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.neighbors import NearestNeighbors
import numpy as np

from torch_geometric.nn import GENConv, LayerNorm, Linear

def get_sinusoidal_encoding(coords, num_freqs=64):
    """
    coords: (N, 2) integer tensor with (y, x) positions
    returns: (N, 4 * num_freqs) tensor of sinusoidal embeddings
    """
    device = coords.device
    N = coords.size(0)
    freqs = torch.arange(num_freqs, dtype=torch.float32, device=device)  # [0, 1, ..., num_freqs-1]
    freqs = 1.0 / (10000 ** (freqs / num_freqs))  # shape: [num_freqs]

    pos_y = coords[:, 0].unsqueeze(1).float()  # [N, 1]
    pos_x = coords[:, 1].unsqueeze(1).float()  # [N, 1]

    y_embed = pos_y * freqs  # [N, num_freqs]
    x_embed = pos_x * freqs

    sin_cos_y = torch.cat([y_embed.sin(), y_embed.cos()], dim=1)
    sin_cos_x = torch.cat([x_embed.sin(), x_embed.cos()], dim=1)
    return torch.cat([sin_cos_y, sin_cos_x], dim=1)  # [N, 4 * num_freqs]

class GraphBranch(nn.Module):
    """Node-level GNN classifier using 3-layer GENConv with edge_attr"""
    def __init__(self, in_channels=512, hidden_channels=128, edge_dim=256, num_layers=3, num_classes=5, dropout=0.1):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GENConv(in_channels, hidden_channels, edge_dim=edge_dim))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
            self.norms.append(LayerNorm(hidden_channels))

        self.convs.append(GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
        self.norms.append(LayerNorm(hidden_channels))

        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, num_classes)
        )

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.convs[0](x, edge_index, edge_attr)
        x = self.norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(1, len(self.convs) - 1):
            residual = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x + residual)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_attr)
        x = self.norms[-1](x)
        x = F.relu(x)

        node_pred = self.classifier(x)
        return node_pred, x  # logits, final embeddings

class GraphHoverNet(nn.Module):
    def __init__(self, num_classes=6, encoder_name='resnet34', encoder_weights='imagenet', 
                 use_graph=True, k_neighbors=8, graph_feature_dim=288,
                 use_vig: bool = True, vig_grid_size: int = 32):
        super().__init__()
        self.use_graph = use_graph
        self.k_neighbors = k_neighbors
        self.use_vig = use_vig
        self.vig_grid_size = vig_grid_size

        # Shared encoder
        self.shared_encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)

        # Use shared encoder in all heads
        self.model_np = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                                         in_channels=3, classes=2, activation=None)
        self.model_hv = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                                         in_channels=3, classes=2, activation=None)
        self.model_nc = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                                         in_channels=3, classes=num_classes, activation=None)

        self.model_np.encoder = self.shared_encoder
        self.model_hv.encoder = self.shared_encoder
        self.model_nc.encoder = self.shared_encoder

        if self.use_graph:
            self.graph_branch = GraphBranch(in_channels=graph_feature_dim,
                                            num_layers=2, num_classes=num_classes)

    def set_stage(self, stage):
        if stage == 'pretrain':
            self.use_graph = False

            for m in [self.model_np, self.model_hv, self.model_nc]:
                    for p in m.parameters():
                        p.requires_grad = True

        elif stage == 'finetune':
            self.use_graph = True

            for p in self.shared_encoder.parameters():
                p.requires_grad = False

            # Unfreeze everything else
            for m in [self.model_np.decoder, self.model_np.segmentation_head,
                      self.model_hv.decoder, self.model_hv.segmentation_head,
                      self.model_nc.decoder, self.model_nc.segmentation_head,
                      self.graph_branch]:
                for p in m.parameters():
                    p.requires_grad = True
                    
    def extract_nucleus_centroids(self, np_pred, hv_pred, threshold=0.5):
        batch_size = np_pred.shape[0]
        centroids_list = []
        for b in range(batch_size):
            prob_map = torch.sigmoid(np_pred[b, 1])
            mask = prob_map > threshold
            if mask.sum() == 0:
                centroids_list.append(torch.empty(0, 2).to(np_pred.device))
                continue
            coords = torch.nonzero(mask, as_tuple=False).float()
            if coords.shape[0] > 500:
                probs = prob_map[mask]
                _, top_indices = torch.topk(probs, 500)
                coords = coords[top_indices]
            centroids_list.append(coords)
        return centroids_list

    def build_graph(self, centroids, features):
        if centroids.shape[0] == 0:
            return torch.empty(2, 0).long().to(centroids.device)
        centroids_np = centroids.cpu().numpy()
        if centroids_np.shape[0] <= self.k_neighbors:
            n = centroids_np.shape[0]
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.extend([[i, j], [j, i]])
            if len(edges) == 0:
                return torch.empty(2, 0).long().to(centroids.device)
            edge_index = torch.tensor(edges).t().contiguous().to(centroids.device)
        else:
            nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, centroids_np.shape[0]))
            nbrs.fit(centroids_np)
            distances, indices = nbrs.kneighbors(centroids_np)
            edges = []
            for i in range(indices.shape[0]):
                for j in range(1, indices.shape[1]):
                    edges.extend([[i, indices[i, j]], [indices[i, j], i]])
            edge_index = torch.tensor(edges).t().contiguous().to(centroids.device)
        return edge_index

    def forward(self, x):
        batch_size = x.shape[0]
        out_np = self.model_np(x)
        out_hv = self.model_hv(x)
        out_nc = self.model_nc(x)
        if not self.use_graph:
            return out_np, out_hv, out_nc, None, None

        out_gc = None
        graph_enhanced_nc = out_nc.clone()
        encoder_feats = self.shared_encoder(x)

        if self.use_vig:
            # Build Visual Interaction Graph (ViG) on a fixed grid from encoder features
            C = encoder_feats[-2].shape[1]
            G = self.vig_grid_size
            small_feats = F.interpolate(encoder_feats[-2], size=(G, G), mode="bilinear", align_corners=False)

            all_coords = []
            all_out_gc = []

            yy, xx = torch.meshgrid(torch.arange(G, device=x.device), torch.arange(G, device=x.device), indexing='ij')
            grid_coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [N, 2]
            Nn = grid_coords.shape[0]

            # Precompute 8-neighborhood edges on grid
            base_idx = (yy * G + xx).reshape(-1)
            edges = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny = yy + dy
                    nx = xx + dx
                    mask = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)
                    src = base_idx[mask]
                    dst = (ny[mask] * G + nx[mask]).reshape(-1)
                    if src.numel() > 0:
                        edges.append(torch.stack([src, dst], dim=0))
            if len(edges) > 0:
                edge_index_template = torch.cat(edges, dim=1)
            else:
                edge_index_template = torch.empty(2, 0, device=x.device, dtype=torch.long)

            for b in range(batch_size):
                node_feats_enc = small_feats[b].permute(1, 2, 0).reshape(Nn, C)  # [N, C]
                pos_enc = get_sinusoidal_encoding(grid_coords, num_freqs=8)  # [N, 32]
                node_features = torch.cat([node_feats_enc, pos_enc], dim=1)  # [N, C+32]

                edge_index = edge_index_template
                if edge_index.shape[1] == 0:
                    continue
                mid = ((grid_coords[edge_index[0]] + grid_coords[edge_index[1]]) // 2).long()
                mid[:, 0] = mid[:, 0].clamp(0, G - 1)
                mid[:, 1] = mid[:, 1].clamp(0, G - 1)
                edge_attr = small_feats[b][:, mid[:, 0], mid[:, 1]].t()  # [E, C]

                out_gc_b, _ = self.graph_branch(node_features, edge_index, edge_attr)
                all_coords.append(torch.cat([grid_coords, torch.full((Nn, 1), b, device=x.device)], dim=1))
                all_out_gc.append(out_gc_b)

            if all_coords:
                centroid_coords = torch.cat(all_coords, dim=0)
                out_gc = torch.cat(all_out_gc, dim=0)
            else:
                centroid_coords = torch.empty(0, 3, device=x.device)
                out_gc = None

            return out_np, out_hv, out_nc, centroid_coords, out_gc
        else:
            # Original nucleus-centroid graph
            encoder_features = F.interpolate(encoder_feats[-2], size=(255, 255), mode="bilinear", align_corners=False)

            all_centroids = []
            all_out_gc = []
            
            for b in range(batch_size):
                centroids = self.extract_nucleus_centroids(out_np[b:b+1], out_hv[b:b+1], threshold=0.3)[0]
                if centroids.shape[0] == 0:
                    continue
                h, w = encoder_features.shape[2], encoder_features.shape[3]
                centroid_coords = centroids.long()
                centroid_coords[:, 0] = torch.clamp(centroid_coords[:, 0], 0, h - 1)
                centroid_coords[:, 1] = torch.clamp(centroid_coords[:, 1], 0, w - 1)
                node_feats = encoder_features[b, :, centroid_coords[:, 0], centroid_coords[:, 1]].t()
                
                # Generate sinusoidal positional encodings
                pos_enc = get_sinusoidal_encoding(centroid_coords, num_freqs=8)  # [N, 32]
                
                # Concatenate encoder + positional features
                node_features = torch.cat([node_feats, pos_enc], dim=1)  # [N, C + 32]

                edge_index = self.build_graph(centroids, node_features)

                mid = ((centroid_coords[edge_index[0]] + centroid_coords[edge_index[1]]) / 2).long()
                mid[:, 0] = mid[:, 0].clamp(0, 254)
                mid[:, 1] = mid[:, 1].clamp(0, 254)
                
                # Edge attributes from midpoints
                edge_attr = encoder_features[b][:, mid[:, 0], mid[:, 1]].t()  # [num_edges, C]
                if edge_index.shape[1] > 0:
                    out_gc, _ = self.graph_branch(node_features, edge_index, edge_attr)
                    all_centroids.append(torch.cat([centroid_coords, torch.full((centroid_coords.shape[0], 1), b, device=x.device)], dim=1))  # (N, 3): (y, x, batch_id)
                    all_out_gc.append(out_gc)

            if all_centroids:
                centroid_coords = torch.cat(all_centroids, dim=0)  # (sum(N), 3)
                out_gc = torch.cat(all_out_gc, dim=0)              # (sum(N), C)
            else:
                centroid_coords = torch.empty(0, 3, device=x.device)
                out_gc = None

            return out_np, out_hv, out_nc, centroid_coords, out_gc

    def print_model_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
    
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Shared encoder parameters: {count_params(self.shared_encoder):,}")
        print(f"NP head parameters (excluding encoder): {count_params(self.model_np.decoder) + count_params(self.model_np.segmentation_head):,}")
        print(f"HV head parameters (excluding encoder): {count_params(self.model_hv.decoder) + count_params(self.model_hv.segmentation_head):,}")
        print(f"NC head parameters (excluding encoder): {count_params(self.model_nc.decoder) + count_params(self.model_nc.segmentation_head):,}")
        print(f"Graph branch parameters: {count_params(self.graph_branch):,}")

if __name__ == "__main__":
    from config import Config
    from loss_function import HoverLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphHoverNet(num_classes=Config.NUM_CLASSES)
    model = model.to(device)
    criterion = HoverLoss()

    print(model)
    print(criterion)