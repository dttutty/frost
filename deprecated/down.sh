#!/bin/bash
       
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt     
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv                
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv                  
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt
# node_np, edge_np = load_feat('GDELT') //413 182

wget -P ./DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/edges.csv        
# node_np, edge_np = load_feat('LASTFM') // 0 0
# Every edge has a matching reverse: likely generated with add_reverse=True
wget -P ./DATA/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/labels.csv          
wget -P ./DATA/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv    
wget -P ./DATA/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/node_features.pt
# node_np, edge_np = load_feat('MAG') // 0 0 

wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv   
# node_np, edge_np = load_feat('MOOC')// 0 0
# ✔ Every edge has a matching reverse: likely generated with add_reverse=True
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
# node_np, edge_np = load_feat('REDDIT') // 0 172
# ✔ Every edge has a matching reverse: likely generated with add_reverse=True

wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv  
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv
# node_np, edge_np = load_feat('WIKI') // 0 172
# ✔ Every edge has a matching reverse: likely generated with add_reverse=True

# python gen_graphs.py --dataset GDELT
# python gen_graphs.py --dataset LASTFM
# python gen_graphs.py --dataset MAG
# python gen_graphs.py --dataset MOOC
# python gen_graphs.py --dataset REDDIT
# python gen_graphs.py --dataset WIKI

node_np, edge_np = load_feat('GDELT') //413 182
_, node_feat_dim, _, _, edge_feat_dim, _ = get_node_edge_dimensions(node_np, edge_np)
logger.debug(node_feat_dim, edge_feat_dim)
