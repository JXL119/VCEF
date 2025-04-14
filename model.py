import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterHead(nn.Module):
    def __init__(self, output_dim, num_prototypes, temperature=10):
        super().__init__()
        self.t = temperature
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        self.normalize_prototypes()
        return self.prototypes(x) * self.t


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5]):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])
        self.weight_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            for _ in kernel_sizes
        ])
        self.out_channels = len(kernel_sizes) * out_channels

    def forward(self, x):
        weighted_features = []
        for branch, weight_layer in zip(self.branches, self.weight_layers):
            feature = branch(x)
            weight = weight_layer(feature)
            weighted_feature = feature * weight
            weighted_features.append(weighted_feature)
        return torch.cat(weighted_features, dim=1)



class WTALModel(nn.Module):
    def __init__(self, config):
        super(WTALModel, self).__init__()
        self.len_feature = config.len_feature
        self.num_classes = config.num_classes
        self.num_clusters = config.num_clusters

        # video classification branch = feature encoder + snippet classifier
        self.cas_module_rgb = nn.Sequential(
            MultiScaleFeatureExtractor(self.len_feature // 2, 512),
            nn.Conv1d(in_channels=512 * 3, out_channels=self.num_classes, kernel_size=1, padding=0),
        )
        self.cas_module_flow = nn.Sequential(
            MultiScaleFeatureExtractor(self.len_feature // 2, 512),
            nn.Conv1d(in_channels=512 * 3, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        # attention branch
        # feature encoder
        self.base_module_rgb = MultiScaleFeatureExtractor(self.len_feature // 2, 512)
        self.base_module_flow = MultiScaleFeatureExtractor(self.len_feature // 2, 512)

        # attention layer
        self.att_module_rgb = nn.Conv1d(in_channels=512 * 3, out_channels=1, kernel_size=1, padding=0)
        self.att_module_flow = nn.Conv1d(in_channels=512 * 3, out_channels=1, kernel_size=1, padding=0)

        # cluster head
        self.clu_head_flow = ClusterHead(512 * 3, self.num_clusters)
        self.clu_head_rgb = ClusterHead(512 * 3, self.num_clusters)

    def forward(self, inp):
        inp = inp.permute(0, 2, 1)

        cas_flow = self.cas_module_flow(inp[:, self.len_feature // 2:, :]).permute(0, 2, 1)
        cas_rgb = self.cas_module_rgb(inp[:, :self.len_feature // 2, :]).permute(0, 2, 1)

        base_flow = self.base_module_flow(inp[:, self.len_feature // 2:, :])
        base_rgb = self.base_module_rgb(inp[:, :self.len_feature // 2, :])

        att_flow = torch.sigmoid(self.att_module_flow(base_flow))
        att_rgb = torch.sigmoid(self.att_module_rgb(base_rgb))

        emb_flow = F.normalize(base_flow.permute(0, 2, 1), dim=-1)
        emb_rgb = F.normalize(base_rgb.permute(0, 2, 1), dim=-1)

        clu_flow = self.clu_head_flow(emb_flow)
        clu_rgb = self.clu_head_rgb(emb_rgb)

        if self.training:
            return (cas_flow, cas_rgb), (att_flow, att_rgb), (clu_flow, clu_rgb), (base_flow, base_rgb)
        else:
            return (cas_flow, cas_rgb), (att_flow, att_rgb), (clu_flow, clu_rgb)
