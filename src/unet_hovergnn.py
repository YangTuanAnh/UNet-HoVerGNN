import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

from torch_geometric.nn import GATv2Conv, LayerNorm, Linear

# Optional SAM import
try:
    from segment_anything import sam_model_registry  # type: ignore
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False

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
    """Node-level GNN classifier using multi-layer GATv2 with edge_attr support."""
    def __init__(self, in_channels=512, hidden_channels=128, edge_dim=256, num_layers=3, num_classes=5, dropout=0.1):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Use single-head GATv2Conv layers to preserve channel sizes and support edge_attr
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=1, concat=True, edge_dim=edge_dim))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=True, edge_dim=edge_dim))
            self.norms.append(LayerNorm(hidden_channels))

        self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=True, edge_dim=edge_dim))
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

class ViTEncoder(nn.Module):
    """
    ViT-style encoder with 4 transformer stages producing hierarchical feature maps.
    Each stage downsamples by 2 (via strided conv) and applies 1 TransformerEncoder layer.
    Returns a list: [input_image, stage1, stage2, stage3, stage4]
    """
    def __init__(self, in_channels: int = 3, embed_dims: list[int] | None = None,
                 num_heads: list[int] | None = None, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        if embed_dims is None:
            embed_dims = [128, 192, 256, 256]
        if num_heads is None:
            num_heads = [4, 6, 8, 8]

        assert len(embed_dims) == 4 and len(num_heads) == 4, "ViTEncoder expects 4 stages"

        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.proj_convs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.pos_projs = nn.ModuleList()

        prev_c = in_channels
        for dim, heads in zip(embed_dims, num_heads):
            # Downsample and project channels
            self.proj_convs.append(nn.Conv2d(prev_c, dim, kernel_size=3, stride=2, padding=1))

            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.transformers.append(nn.TransformerEncoder(layer, num_layers=1))
            self.pos_projs.append(Linear(32, dim))
            prev_c = dim

        # Expose channels for heads
        self.out_channels = [in_channels] + embed_dims

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = [x]
        cur = x
        B = x.shape[0]
        device = x.device
        for proj, tr, pos_lin in zip(self.proj_convs, self.transformers, self.pos_projs):
            cur = proj(cur)
            B_, C, H, W = cur.shape
            tokens = cur.flatten(2).transpose(1, 2)  # [B, N, C]
            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).to(torch.long)
            pos = get_sinusoidal_encoding(coords, num_freqs=8)  # [N, 32]
            pos = pos_lin(pos).unsqueeze(0).expand(B_, -1, -1)
            tokens = tr(tokens + pos)
            cur = tokens.transpose(1, 2).reshape(B_, C, H, W)
            feats.append(cur)
        return feats

class SAMViTEncoder(nn.Module):
    """
    Wraps SAM's image encoder to provide a 4-level feature pyramid for UNet decoders.
    - Loads SAM ViT (vit_b/l/h) from checkpoint via segment_anything
    - Produces [x0, s1, s2, s3, s4] where s1 is the SAM image embedding, and s2..s4 are
      progressively downsampled via strided convs.
    """
    def __init__(self, sam_model_type: str = 'vit_b', sam_checkpoint: str | None = None,
                 out_channels: list[int] | None = None, normalize: bool = True):
        super().__init__()
        assert _HAS_SAM, "segment_anything is not installed. Install it or disable SAM encoder."
        assert sam_checkpoint is not None and len(sam_checkpoint) > 0, "sam_checkpoint path must be provided."
        self.normalize = normalize
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        # Expect SAM image_encoder attribute
        self.image_encoder = self.sam.image_encoder

        # Default output channels to match our decoders
        if out_channels is None:
            # [input, s1, s2, s3, s4]
            out_channels = [3, 128, 192, 256, 256]
        assert len(out_channels) == 5
        self.out_channels = out_channels

        # Project SAM embedding channels to our s1 channels if needed
        # Infer SAM embed dim by a dummy spec (can't run here), so define a lazy conv after first forward
        self.s1_proj: nn.Module | None = None

        # Downsampling towers to create s2..s4
        c1, c2, c3, c4 = out_channels[1], out_channels[2], out_channels[3], out_channels[4]
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.GELU(),
        )

        # SAM normalization parameters (from official repo)
        self.register_buffer('pixel_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255.0, persistent=False)
        self.register_buffer('pixel_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1) / 255.0, persistent=False)

    @staticmethod
    def _to_multiple_of_16(h: int, w: int) -> tuple[int, int]:
        import math
        return int(math.ceil(h / 16) * 16), int(math.ceil(w / 16) * 16)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, C, H, W = x.shape
        feats: list[torch.Tensor] = [x]

        # Prepare input for SAM: resize to multiple of 16 and normalize
        Hs, Ws = self._to_multiple_of_16(H, W)
        x_resized = x
        if (Hs != H) or (Ws != W):
            x_resized = F.interpolate(x_resized, size=(Hs, Ws), mode='bilinear', align_corners=False)
        if self.normalize:
            x_resized = (x_resized - self.pixel_mean) / self.pixel_std

        # SAM image encoder forward: returns [B, Cs, Hs/16, Ws/16]
        with torch.no_grad():
            sam_feat = self.image_encoder(x_resized)

        # Project to s1 channels if needed
        target_c1 = self.out_channels[1]
        if self.s1_proj is None:
            in_c = sam_feat.shape[1]
            if in_c != target_c1:
                self.s1_proj = nn.Conv2d(in_c, target_c1, kernel_size=1)
            else:
                self.s1_proj = nn.Identity()
        s1 = self.s1_proj(sam_feat)

        # Build pyramid s2..s4
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)

        feats.extend([s1, s2, s3, s4])
        return feats

class UNetSegHead(nn.Module):
    """
    CNN-based UNet decoder head that consumes encoder pyramid [input, s1..s4].
    Fuses deepest->shallowest with skip connections, then fuses with input skip.
    """
    def __init__(self, encoder_channels: list[int], num_classes: int, embed_dim: int = 256):
        super().__init__()
        assert len(encoder_channels) >= 5, "Expect [input, s1, s2, s3, s4] from encoder"

        self.num_classes = num_classes

        # Channels at each encoder stage
        c0, c1, c2, c3, c4 = encoder_channels[:5]

        # Decoder channels per upsampling step (from deep to shallow)
        decoder_channels = [embed_dim, max(embed_dim // 2, 96), max(embed_dim // 4, 64)]

        # Blocks: s4 -> s3, s3 -> s2, s2 -> s1
        self.block1 = nn.Sequential(
            nn.Conv2d(c4, decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(decoder_channels[0] + c3, decoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.GELU(),
            nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.GELU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(decoder_channels[1] + c2, decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.GELU(),
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.GELU(),
        )

        # After reaching s1, fuse with s1 skip
        self.block4 = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + c1, decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.GELU(),
        )

        # Input skip projection and final fuse
        self.input_proj = nn.Conv2d(c0, decoder_channels[2], kernel_size=1)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + decoder_channels[2], decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.GELU(),
        )

        self.seg_head = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)

    def forward(self, encoder_feats: list[torch.Tensor], input_spatial_size: tuple[int, int]) -> torch.Tensor:
        B = encoder_feats[0].shape[0]
        H, W = input_spatial_size

        x0, x1, x2, x3, x4 = encoder_feats[:5]

        # Start from deepest
        x = self.block1(x4)
        x = F.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.block2(x)

        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.block3(x)

        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.block4(x)

        # Fuse with input skip
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        input_feat = self.input_proj(x0)
        input_feat = F.interpolate(input_feat, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat([x, input_feat], dim=1)
        x = self.final_fuse(x)

        logits = self.seg_head(x)
        return logits

class GraphHoverNet(nn.Module):
    def __init__(self, num_classes=6, use_graph=True, k_neighbors=8,
                 use_sam_encoder: bool = False, sam_model_type: str = 'vit_b', sam_checkpoint: str | None = None,
                 graph_knn_backend: str = 'torch', use_amp: bool = False):
        super().__init__()
        self.use_graph = use_graph
        self.k_neighbors = k_neighbors
        self.graph_knn_backend = graph_knn_backend  # 'torch' or 'sklearn'
        self.use_amp = use_amp

        # Encoder choice: SAM-based ViT or local ViTEncoder
        if use_sam_encoder:
            assert _HAS_SAM, "segment_anything is not available. Install it or set use_sam_encoder=False."
            assert sam_checkpoint is not None and len(sam_checkpoint) > 0, "Provide sam_checkpoint when use_sam_encoder=True."
            self.encoder = SAMViTEncoder(sam_model_type=sam_model_type, sam_checkpoint=sam_checkpoint)
        else:
            self.encoder = ViTEncoder(in_channels=3)
        self.encoder_channels = list(self.encoder.out_channels)

        # Graph uses the -2 stage (s3)
        self.feature_stage_index = -2
        self.feature_channels = int(self.encoder_channels[self.feature_stage_index])

        # CNN-based UNet decoder heads for NP, HV, and NC tasks
        self.head_np = UNetSegHead(encoder_channels=self.encoder_channels, num_classes=2, embed_dim=256)
        self.head_hv = UNetSegHead(encoder_channels=self.encoder_channels, num_classes=2, embed_dim=256)
        self.head_nc = UNetSegHead(encoder_channels=self.encoder_channels, num_classes=num_classes, embed_dim=256)

        if self.use_graph:
            # Graph in_channels = feature_channels + positional_dim (32)
            computed_graph_in = self.feature_channels + 32
            self.graph_branch = GraphBranch(in_channels=computed_graph_in,
                                            hidden_channels=128,
                                            edge_dim=self.feature_channels,
                                            num_layers=2, num_classes=num_classes)

    def set_stage(self, stage):
        if stage == 'pretrain':
            self.use_graph = False

            # Train ViT encoder and CNN decoders
            for p in self.encoder.parameters():
                p.requires_grad = True
            for m in [self.head_np, self.head_hv, self.head_nc]:
                for p in m.parameters():
                    p.requires_grad = True

        elif stage == 'finetune':
            self.use_graph = True

            # Freeze ViT encoder, finetune decoder heads and graph branch
            for p in self.encoder.parameters():
                p.requires_grad = False

            modules_to_train = [self.head_np, self.head_hv, self.head_nc]
            if hasattr(self, 'graph_branch'):
                modules_to_train.append(self.graph_branch)

            for m in modules_to_train:
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
        if self.graph_knn_backend == 'torch':
            return self._build_graph_torch(centroids)
        else:
            return self._build_graph_sklearn(centroids)

    def _build_graph_torch(self, centroids: torch.Tensor) -> torch.Tensor:
        # centroids: [N, 2] on device
        N = centroids.shape[0]
        device = centroids.device
        if N <= self.k_neighbors:
            if N < 2:
                return torch.empty(2, 0, device=device, dtype=torch.long)
            idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=device)
            edges = torch.stack([torch.cat([idx_i, idx_j], dim=0), torch.cat([idx_j, idx_i], dim=0)], dim=0)
            return edges.long()
        # Compute pairwise distances (squared euclidean) and take top-k
        coords = centroids.float()
        dists = torch.cdist(coords, coords, p=2)
        # Exclude self by setting large value on diagonal
        diag_idx = torch.arange(N, device=device)
        dists[diag_idx, diag_idx] = float('inf')
        k = min(self.k_neighbors, N - 1)
        knn_dists, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn_idx.reshape(-1)
        edges = torch.stack([src, dst], dim=0)
        # Make undirected by adding reverse edges (already included by construction, but ensure symmetry)
        edges_rev = torch.stack([edges[1], edges[0]], dim=0)
        edge_index = torch.cat([edges, edges_rev], dim=1)
        return edge_index.long()

    def _build_graph_sklearn(self, centroids: torch.Tensor) -> torch.Tensor:
        centroids_np = centroids.detach().cpu().numpy()
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

        amp_ctx = torch.cuda.amp.autocast if (self.use_amp and x.is_cuda) else torch.cpu.amp.autocast
        with amp_ctx(enabled=self.use_amp):
            # Run ViT encoder once
            encoder_feats = self.encoder(x)
            feat = encoder_feats[self.feature_stage_index]  # [B, C, Hf, Wf]

            # CNN UNet-style decoder heads
            out_np = self.head_np(encoder_feats, input_spatial_size=(x.shape[2], x.shape[3]))
            out_hv = self.head_hv(encoder_feats, input_spatial_size=(x.shape[2], x.shape[3]))
            out_nc = self.head_nc(encoder_feats, input_spatial_size=(x.shape[2], x.shape[3]))

        if not self.use_graph:
            return out_np, out_hv, out_nc, None, None

        out_gc = None
        encoder_features = F.interpolate(feat, size=(255, 255), mode="bilinear", align_corners=False)

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
            
            # Edge attributes from midpoints (use feature channels)
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
        print(f"ViT encoder parameters: {count_params(self.encoder):,}")
        print(f"NP UNet head parameters: {count_params(self.head_np):,}")
        print(f"HV UNet head parameters: {count_params(self.head_hv):,}")
        print(f"NC UNet head parameters: {count_params(self.head_nc):,}")
        if hasattr(self, 'graph_branch'):
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