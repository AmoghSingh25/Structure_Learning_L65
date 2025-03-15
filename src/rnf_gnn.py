import argparse
import bnlearn as bn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from dgl import DGLGraph
from modules import GAT_RNF


#### Setting up argparse ####
parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', type=str)
parser.add_argument('-num_samples', type=int)
parser.add_argument('-rnf_dim', type=int)
parser.add_argument('-rnf_intermediate_dim', type=int)
parser.add_argument('-rnf_init_method', type=str)
parser.add_argument('-num_heads', type=int)
parser.add_argument('-num_epochs', type=int)
parser.add_argument('-learning_rate', type=float)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-experiment_id', type=str)
args = parser.parse_args()

#### Defining the paths ####
dataset_path = f"BIFs/{args.dataset_name}.bif"

save_dir = f"results/{args.experiment_id}"
#### Building the dataset ####
causal_model = bn.import_DAG(dataset_path)
df_samples = bn.sampling(causal_model, n=args.num_samples)
nodes_orig = list(causal_model['adjmat'].columns)
num_nodes = len(nodes_orig)

target_symmetric_adj = (causal_model['adjmat'] + causal_model['adjmat'].T).to_numpy()

node_features = torch.from_numpy(df_samples.to_numpy()).type(torch.float32).unsqueeze(2)

#### Defining functionality ####
def build_graph(features, num_nodes, src_nodes, dst_nodes):
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.ndata['features'] = features
    g.add_edges(src_nodes, dst_nodes)
    return g

def collate_fn(data):
    return data

def train_epoch(loader, model, optimizer, loss_fn):
    for g_list in loader:
        optimizer.zero_grad()
        target_ls, pred_ls = [], []
        for g in g_list:
            target_ls.append(g.ndata['features'])
            out = model(g, g.ndata['features'])
            pred_ls.append(out)
        loss = loss_fn(torch.stack(tuple(out)), torch.stack(tuple(target_ls)))
        loss.backward()
        optimizer.step()

#### Preparing for training ####
initial_adj = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
src_nodes, dst_nodes = torch.nonzero(initial_adj).T

graph_dataset = [build_graph(features, num_nodes, src_nodes, dst_nodes) for features in node_features]

loader = DataLoader(graph_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

#### Main Operations
model = GAT_RNF(num_nodes=num_nodes, rnf_dim=args.rnf_dim, rnf_intermediate_dim=args.rnf_intermediate_dim, rnf_init_method=args.rnf_init_method, num_heads=args.num_heads)

optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()

for epoch_id in range(1,args.num_epochs+1):
    print(f'Epoch {epoch_id}')
    train_epoch(loader, model, optimizer, loss_fn)
torch.save(model.state_dict(), "saved_model/gat_rnf.pt")