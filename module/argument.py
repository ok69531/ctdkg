import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cd', help='dataset name, default to cd (cd, cg-v1, cg-v2, gd, cgd, cgpd, ctd)')
    parser.add_argument('--model', default='TransE', type=str, help='TransE, RotatE, HAKE, GIE, HOUSE, TripleRE, DistMult, ComplEx, conve, rgcn, compgcn')
    parser.add_argument('--num_entity_embedding', default=1, type=int)
    parser.add_argument('--num_relation_embedding', default=1, type=int)
    
    parser.add_argument('--negative_sample_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--gamma', default=4.0, type=float)
    parser.add_argument('--negative_adversarial_sampling', default=True)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', default=True, 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('--negative_loss', action='store_true', 
                        help='use negative sampling loss to train ConvE')
    # ---- GIE options ---- #
    parser.add_argument("--init_size", default=1e-3, type=float, help="Initial embeddings' scale")
    parser.add_argument("--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)")
    # ---- HousE options ---- #
    parser.add_argument('--house_dim', default=2, type=int)
    parser.add_argument('--housd_num', default=1, type=int)
    parser.add_argument('--thred', default=0.5, type=float)
    
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--num_workers', default=4, type = int)
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--max_step', default = 100000, type = int)
    # parser.add_argument('--num_epoch', default = 100, type = int)
    
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--test_log_steps', default=3000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--use_description', type = bool, default = False, help = 'whether using the text description')
    
    return parser.parse_args(args)
