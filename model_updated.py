from torch_geometric.nn import GATConv, SAGEConv, to_hetero
import torch.nn.functional as F
import torch 
from torch import nn
from torch.nn import Linear, LazyLinear
from torch import Tensor
from torch_geometric.data import HeteroData
# from gnn import GNN

# class Linear(nn.Module):
#     def __init__(self, dim_in, dim_out, bias=False, **kwargs):
#         super(Linear, self).__init__()
#         self.model = nn.Linear(dim_in, dim_out, bias=bias)

#     def forward(self, x):
#         return self.model(x)

class GAT_GNN_Edge_Feats(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = Linear(hidden_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels, edge_dim = 768, add_self_loops= False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim = 768, add_self_loops= False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, edge_dim = 768, add_self_loops= False)
        self.conv4 = GATConv(hidden_channels, hidden_channels, edge_dim = 768, add_self_loops= False)
        self.conv5 = GATConv(hidden_channels, hidden_channels, edge_dim = 768, add_self_loops= False)
        self.conv6 = GATConv(hidden_channels, hidden_channels, edge_dim = 768, add_self_loops= False)
        self.linear3 = Linear(hidden_channels*2, out_channels)
        # self.linear4 = Linear(hidden_channels, hidden_channels)
        # self.linear5 = Linear(hidden_channels, out_channels)
        
        #constructing message passing layers
        # self.f = []
        # num_layers = 6
        # for i in range(num_layers - 1):
        #     # d_in = dim_in if i == 0 else dim_out
        #     self.f.append(SAGEConv(hidden_channels, hidden_channels))
        # # d_in = dim_in if num_layers == 1 else dim_out
        # self.f.append(SAGEConv(hidden_channels, hidden_channels, has_act=False))
        # self.f = nn.Sequential(*self.f)
        self.act = nn.ReLU(inplace=False)
        
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr) -> Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        # print("here")
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        x = self.conv5(x, edge_index, edge_attr=edge_attr)
        x = self.conv6(x, edge_index, edge_attr=edge_attr)
        x = torch.cat((x, x), 1)
        x = self.act(x)
        x = self.linear3(x)
        # x = self.linear4(x)
        # x = self.linear5(x)
        return x

class GAT_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = Linear(hidden_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels, add_self_loops= False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops= False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, add_self_loops= False)
        self.conv4 = GATConv(hidden_channels, hidden_channels, add_self_loops= False)
        self.conv5 = GATConv(hidden_channels, hidden_channels, add_self_loops= False)
        self.conv6 = GATConv(hidden_channels, hidden_channels, add_self_loops= False)
        self.linear3 = Linear(hidden_channels*2, out_channels)
        # self.linear4 = Linear(hidden_channels, hidden_channels)
        # self.linear5 = Linear(hidden_channels, out_channels)
        
        #constructing message passing layers
        # self.f = []
        # num_layers = 6
        # for i in range(num_layers - 1):
        #     # d_in = dim_in if i == 0 else dim_out
        #     self.f.append(SAGEConv(hidden_channels, hidden_channels))
        # # d_in = dim_in if num_layers == 1 else dim_out
        # self.f.append(SAGEConv(hidden_channels, hidden_channels, has_act=False))
        # self.f = nn.Sequential(*self.f)
        self.act = nn.ReLU(inplace=False)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        # print("here")
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = self.conv6(x, edge_index)
        x = torch.cat((x, x), 1)
        x = self.act(x)
        x = self.linear3(x)
        # x = self.linear4(x)
        # x = self.linear5(x)
        return x  

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = Linear(hidden_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.conv5 = SAGEConv(hidden_channels, hidden_channels)
        self.conv6 = SAGEConv(hidden_channels, hidden_channels)
        self.linear3 = Linear(hidden_channels*2, out_channels)

        self.act = nn.ReLU(inplace=False)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        # print("here")
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = self.conv6(x, edge_index)
        x = torch.cat((x, x), 1)
        x = self.act(x)
        x = self.linear3(x)
        # x = self.linear4(x)
        # x = self.linear5(x)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_comm1: Tensor, x_comm2: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_comm1 = x_comm1[edge_label_index[0]].T
        edge_feat_comm2 = x_comm2[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_comm1 * edge_feat_comm2).sum(dim=-1)

class proj_CommunityArticleGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        self.article_proj = 1024
        self.comm_proj = 1024
        
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        
        self.lin1 = LazyLinear(self.article_proj)
        self.lin2 = LazyLinear(self.comm_proj)
        self.lin3 = Linear(hidden_channels, out_channels)

    #attends over both mentioned by and written by relations
    def forward(self, data):
        
        article_x = self.lin1(data['article'].x).reshape((self.article_proj, 1))
        community_x = self.lin2(data['community'].x).reshape((self.comm_proj, 1))
        
        art_comm_h = self.conv1(
            (article_x, community_x),
            data['article', 'written_by', 'community'].edge_index,
        ).relu()
        
        art_comm_h_prime = self.conv2(
            (article_x, art_comm_h),
            data['article', 'mentioned_by', 'community'].edge_index,
        ).relu()
        
        community_h = self.conv3(
            (art_comm_h_prime, data['community'].x),
            data['community', 'interacts_with', 'community'].edge_index,
        ).relu()
        
        return self.lin3(community_h)

class Abl_CommunityArticleGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        self.article_proj = 1024
        
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        
        self.lin1 = LazyLinear(self.article_proj)
        self.lin2 = Linear(hidden_channels, out_channels)

    #attends over both mentioned by and written by relations
    def forward(self, data, comm_emb, article_emb, ablation_type="community"):
        if ablation_type == "community":
            community_data = comm_emb
            article_data = data['article'].x
        elif ablation_type == "article":
            community_data = data['community'].x
            article_data = article_emb
        elif ablation_type == "comm_article":
            community_data = comm_emb
            article_data = article_emb   
        else:
            raise ValueError("ablation_type must be one of 'community', 'article', or 'comm_article'")
        # print(community_data.shape)
        # print(article_data.shape)
        article_x = self.lin1(article_data).reshape((self.article_proj, 1))
        art_comm_h = self.conv1(
            (article_x, community_data),
            data['article', 'written_by', 'community'].edge_index,
        ).relu()
        
        art_comm_h_prime = self.conv2(
            (article_x, art_comm_h),
            data['article', 'mentioned_by', 'community'].edge_index,
        ).relu()
        
        community_h = self.conv3(
            (art_comm_h_prime, data['community'].x),
            data['community', 'interacts_with', 'community'].edge_index,
        ).relu()
        
        return self.lin2(community_h)


class CommunityArticleGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        self.article_proj = 1024
        
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        
        self.lin1 = LazyLinear(self.article_proj)
        self.lin2 = Linear(hidden_channels, out_channels)

    #attends over both mentioned by and written by relations
    def forward(self, data):
        
        article_x = self.lin1(data['article'].x).reshape((self.article_proj, 1))
        art_comm_h = self.conv1(
            (article_x, data['community'].x),
            data['article', 'written_by', 'community'].edge_index,
        ).relu()
        
        art_comm_h_prime = self.conv2(
            (article_x, art_comm_h),
            data['article', 'mentioned_by', 'community'].edge_index,
        ).relu()
        
        community_h = self.conv3(
            (art_comm_h_prime, data['community'].x),
            data['community', 'interacts_with', 'community'].edge_index,
        ).relu()
        
        return self.lin2(community_h)

class CommunityGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels)
        # self.conv2 = GATConv((-1, -1), hidden_channels)
        # self.conv3 = GATConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_indices):
        community_h = self.conv1(
            x,
            edge_indices,
        ).relu()

        return self.lin(community_h)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_label_indices):
        row, col = edge_label_indices
        z = torch.cat([z[row], z[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, modelname, num_comm_nodes, num_article_nodes, device=None):
        super().__init__()
        if device != None:
            self.abl_comm_emb = torch.rand((num_comm_nodes, hidden_channels), requires_grad=True).to(device)
            self.abl_article_emb = torch.rand((num_article_nodes, hidden_channels), requires_grad=True).to(device)
        
        self.comm_encoder = CommunityGNNEncoder(hidden_channels, out_channels)
        self.comm_art_encoder = CommunityArticleGNNEncoder(hidden_channels, out_channels)
        
        self.abl_comm_art_encoder = Abl_CommunityArticleGNNEncoder(hidden_channels, out_channels)
        
        self.proj_comm_art_encoder = proj_CommunityArticleGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)
    
    #article features included
    def forward(self, data):
        community_z = self.comm_art_encoder(data)
        # community_z = self.proj_comm_art_encoder(data)

        return self.decoder(community_z, data['community', 'interacts_with', 'community'].edge_index)

    
    #article features ablated
    def forward_art_ablate(self, data):
        community_z = self.comm_encoder(data['community'].x, data['community', 'interacts_with', 'community'].edge_index)

        return self.decoder(community_z, data['community', 'interacts_with', 'community'].edge_index)
    
    def forward_ablate(self, data, ablation_type="community"):
        community_z = self.abl_comm_art_encoder(data, self.abl_comm_emb, self.abl_article_emb, ablation_type=ablation_type)

        return self.decoder(community_z, data['community', 'interacts_with', 'community'].edge_index)