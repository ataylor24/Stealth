from torch_geometric.nn import GATConv, SAGEConv, to_hetero
import torch.nn.functional as F
import torch 
from torch import nn
from torch import Tensor
from torch_geometric.data import HeteroData
# from gnn import GNN

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return self.model(x)

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
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_comm1: Tensor, x_comm2: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_comm1 = x_comm1[edge_label_index[0]].T
        edge_feat_comm2 = x_comm2[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_comm1 * edge_feat_comm2).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, modelname, num_comm_nodes, num_article_nodes, metadata, embedding_dim):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.comm_lin = torch.nn.Linear(embedding_dim, hidden_channels)
        self.comm_emb = torch.nn.Embedding(num_comm_nodes, hidden_channels)
        self.article_emb = torch.nn.Embedding(num_article_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        if modelname == "graphsage":
            self.edge_attrs = False
            self.gnn = GNN(hidden_channels, out_channels)
        elif modelname == "gat":
            self.edge_attrs = False
            self.gnn = GAT_GNN(hidden_channels, out_channels)
        elif modelname == "gat_edge_feats":
            self.edge_attrs = True
            self.gnn = GAT_GNN_Edge_Feats(hidden_channels, out_channels)
        else:
            raise NotImplementedError
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = Classifier()
        
    def forward(self, device, data: HeteroData) -> Tensor:
        # print(device)
        #UNCOMMENT FOR MEMORY SAVING
        # data["community"].x = data["community"].x.to(device)
        # data["community"].node_id = data["community"].node_id.to(device)
        # data["article"].node_id = data["article"].node_id.to(device)
        # print("data[article].node_id", data["article"].node_id)
        x_dict = {
          "community": self.comm_lin(data["community"].x) + self.comm_emb(data["community"].node_id),
          "article": self.article_emb(data["article"].node_id),
        } 
        edge_index_dict = {}
        for edge_type in data.edge_index_dict:
            # print(edge_index_dict[edge_type].device)
            edge_index_dict[edge_type] = data.edge_index_dict[edge_type].to(device)
        
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        
        if not self.edge_attrs:
            x_dict = self.gnn(x_dict, edge_index_dict)#data.edge_index_dict)
        else:
            x_dict = self.gnn(x_dict, edge_index_dict, data.edge_attrs)#data.edge_index_dict)

        pred = self.classifier(
            x_dict["community"],
            x_dict["community"],
            data["community", "interacts_with", "community"].edge_label_index,
        )
        del x_dict, edge_index_dict, data
        return pred