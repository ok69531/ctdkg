# CTDKG
Source code for 'A New Large-Scale Benchmark Knowledge Graph Leveraging the Comparative Toxicogenomics Database' with pytorch and torch_geometric.


## Requirements & Installation
The code is written in Python 3.10.0 and required packages as follow:
- torch
- torch_geometric
- torch_scatter (not necessary, need for compgcn implementation)

Please check [INSTALL.md](./INSTALL.md) for installation details.


## Basic Usage
### Data download
```python
from module.dataset import LinkPredDataset

# Download and process data at './dataset/cd'
dataset = LinkPredDataset(name = 'cd')
train_triples, valid_triples, test_triples = dataset.get_edge_split()
```
If you have problems downloading the dataset, refer to the [link](https://github.com/ok69531/ctdkg/releases) and manually install the dataset. This will be modified soon.

### Data loader
``` python
# head loader for train data
data = dataset[0]
train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
for i in tqdm(range(len(train_triples['head']))):
    head, relation, tail = train_triples['head'][i].item(), train_triples['relation'][i].item(), train_triples['tail'][i].item()
    head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
    train_count[(head, relation, head_type)] += 1
    train_count[(tail, -relation-1, tail_type)] += 1
    train_true_head[(relation, tail)].append(head)
    train_true_tail[(head, relation)].append(tail)

train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, 
        args.negative_sample_size, 'head-batch',
        train_count, train_true_head, train_true_tail,
        entity_dict), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=args.num_workers,
    collate_fn=TrainDataset.collate_fn
)

# head loader for validation/test data
valid_dataloader_head = DataLoader(
    TestDataset(
        valid_triples, 
        args, 
        'head-batch',
        random_sampling,
        entity_dict
    ),
    batch_size = args.test_batch_size,
    num_workers = args.num_workers,
    collate_fn = TestDataset.collate_fn
)
```
The head loader means that head entities of true triples were contaminated $(h', r, t)$. If you want to make a tail loader, substitute 'head-batch' with 'tail-batch.'
If you run the code on the local environment, then set the num_workers argument to `num_workers=0`.

### Training & Evaluation
We support eight models and seven datasets. 

Datasets: ```cd, cg-v1, cg-v2, gd, cgd, cgpd, ctd```
```python
# for translation models
python main.py --dataset=cd --model=TransE
python main.py --dataset=cd --model=RotatE -double_entity_embedding
python main.py --dataset=cd --model=HAKE --double_entity_embedding --num_relation_embedding=3

# for semantic information models
python main.py --dataset=cd --model=DistMult
python main.py --dataset=cd --model=ComplEx --double_entity_embedding --num_relation_embedding=2

# for neural network models
python main_conv.py --dataset=cd --model=conve --learning_rate=0.001 --negative_sample_size=1
python main_conv.py --dataset=cd --model=convkb --learning_rate=0.005 --negative_sample_size=1
python main_gnn.py --dataset=cd --model=rgcn --learning_rate=0.001 --negative_sample_size=1
```


## Components
```
├── build_dataset
│   ├── data_download.py
│   ├── generate_negative.py
│   ├── preprocess.py
│   ├── scrap_gene_phenotype.py
├── module
│   ├── argument.py
│   ├── dataset.py
│   ├── compgcn_layer.py
│   ├── model.py
│   └── set_seed.py
├── main.py
├── main_gnn.py
├── main_conv.txt
└── .gitignore
```
- build_dataset/data_download.py: downloading CTD data
- build_dataset/generate_negative.py: generating negative samples for validation/test data
- build_dataset/preprocess.py: building benchmark graphs
- build_dataset/scrap_gene_phenotype.py: crawling gene-phenotype data from CTD

- module/argument.py: set of arguments
- module/dataset.py: downloading and loading dataset for training
- module/compgcn_layer.py: compgcn layer
- module/model.py: model architectures
- module/set_seed.py: specify the seed

- main.py: script for training using translation and semantic models
- main_conv.py: script for training using convolution-based models
- main_gnn.py: script for training using GNN-based models

<!-- 
## Tutorial
We provide a tutorial conducted in the Google Colab environment: [link](https://colab.research.google.com/drive/1ePTpkQdWiHQotlXLagbAwbeVVEjRn2BI?usp=sharing) -->

<!-- The tutorial can be divided into three main parts: importing necessary data and models for training, calculating the one-step influence function, and implementing the overall algorithm of our method.

A copy of the Colab page is uploaded to the repository as the [aais_example.ipynb](https://github.com/ok69531/AAIS-public/blob/main/aais_example.ipynb) file. -->
