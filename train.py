# Import the graphscope module

import graphscope

graphscope.set_option(show_log=False)  # enable logging

# Load the obgn_mag dataset as a graph
# Create a session on kubernetes cluster and 
# mount dataset bucket to path "/home/jovyan/datasets" in pod.

from graphscope.dataset import load_ogbn_mag

#sess = graphscope.session(k8s_dataset_image="registry.cn-hongkong.aliyuncs.com/graphscope/dataset:jupyter", mount_dataset="/home/jovyan/datasets")

sess = graphscope.session()

print("SESSION: ", sess)
# Load the obgn_mag dataset in "sess" as a graph

graph = load_ogbn_mag(sess, "/home/jovyan/datasets/ogbn_mag_small/")

# Get the entrypoint for submitting Gremlin queries on graph g.
interactive = sess.gremlin(graph)

# Count the number of papers two authors (with id 2 and 4307) have co-authored.
papers = interactive.execute(
    "g.V().has('author', 'id', 2).out('writes').where(__.in('writes').has('id', 4307)).count()").one()
print("result", papers)
# Exact a subgraph of publication within a time range.
# Exact a subgraph of publication within a time range.
sub_graph = interactive.subgraph(
    "g.V().has('year', inside(2014, 2020)).outE('cites')"
)

# Project the subgraph to simple graph by selecting papers and their citations.
simple_g = sub_graph.project(vertices={"paper": []}, edges={"cites": []})
# compute the kcore and triangle-counting.
kc_result = graphscope.k_core(simple_g, k=5)
tc_result = graphscope.triangles(simple_g)

# Add the results as new columns to the citation graph.
sub_graph = sub_graph.add_column(kc_result, {"kcore": "r"})
sub_graph = sub_graph.add_column(tc_result, {"tc": "r"})
# Define the features for learning, 
# we chose original 128-dimension feature and k-core, triangle count result as new features.
# Define the features for learning, 
# we chose original 128-dimension feature and k-core, triangle count result as new features.
paper_features = []
for i in range(128):
    paper_features.append("feat_" + str(i))
paper_features.append("kcore")
paper_features.append("tc")

# launch a learning engine. here we split the dataset, 75% as train, 10% as validation and 15% as test.
lg = sess.graphlearn(sub_graph, nodes=[("paper", paper_features)],
                     edges=[("paper", "cites", "paper")],
                     gen_labels=[
                         ("train", "paper", 100, (0, 75)),
                         ("val", "paper", 100, (75, 85)),
                         ("test", "paper", 100, (85, 100))
                     ])

# Then we define the training process, use internal GCN model.
import graphscope.learning
from graphscope.learning.examples import GCN
from graphscope.learning.graphlearn.python.model.tf.trainer import LocalTFTrainer
from graphscope.learning.graphlearn.python.model.tf.optimizer import get_tf_optimizer

def train(config, graph):
    def model_fn():
        return GCN(graph,
                    config["class_num"],
                    config["features_num"],
                    config["batch_size"],
                    val_batch_size=config["val_batch_size"],
                    test_batch_size=config["test_batch_size"],
                    categorical_attrs_desc=config["categorical_attrs_desc"],
                    hidden_dim=config["hidden_dim"],
                    in_drop_rate=config["in_drop_rate"],
                    neighs_num=config["neighs_num"],
                    hops_num=config["hops_num"],
                    node_type=config["node_type"],
                    edge_type=config["edge_type"],
                    full_graph_mode=config["full_graph_mode"])
    # graphscope.learning.reset_default_tf_graph()
    trainer = LocalTFTrainer(model_fn,
                             epoch=config["epoch"],
                             optimizer=get_tf_optimizer(
                             config["learning_algo"],
                             config["learning_rate"],
                             config["weight_decay"]))
    trainer.train_and_evaluate()
    
# hyperparameters config.
config = {"class_num": 349, # output dimension
            "features_num": 130, # 128 dimension + kcore + triangle count
            "batch_size": 500,
            "val_batch_size": 100,
            "test_batch_size":100,
            "categorical_attrs_desc": "",
            "hidden_dim": 256,
            "in_drop_rate": 0.5,
            "hops_num": 2,
            "neighs_num": [5, 10],
            "full_graph_mode": False,
            "agg_type": "gcn",  # mean, sum
            "learning_algo": "adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
            "epoch": 5,
            "node_type": "paper",
            "edge_type": "cites"}

# Start traning and evaluating
train(config, lg)