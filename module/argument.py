import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    # parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    # parser.add_argument('--do_train', action='store_true')
    # parser.add_argument('--do_valid', action='store_true')
    # parser.add_argument('--do_test', action='store_true')
    # parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cd', help='dataset name, default to cd (cd, cg-v1, cg-v2, gd, cgd, cgpd, ctd)')
    parser.add_argument('--train_frac', type=float, default=0.1, help='fraction of training data for large scale dataset')
    parser.add_argument('--model', default='TransE', type=str, help='TransE, RotatE, DistMult, ComplEx, conve, rgcn, compgcn')
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    # parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    parser.add_argument('-nr', '--num_relation_embedding', default=1, type=int)
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=100, type=int)
    parser.add_argument('-g', '--gamma', default=4.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=32, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', default=True, 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('-nl', '--negative_loss', action='store_true', 
                        help='use negative sampling loss to train ConvE')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-nw', '--num_workers', default=1, type = int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--num_epoch', default = 100, type = int)
    # parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    # parser.add_argument('-save', '--save_path', default=None, type=str)
    # parser.add_argument('--max_steps', default=300000, type=int)
    # parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--test_log_steps', default=3000, type=int, help='valid/test log every xx steps')
    # parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    # parser.add_argument('--log_steps', default=10, type=int, help='train log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    # parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    # parser.add_argument('--ntriples_eval_train', type=int, default=200000, help='number of training triples to evaluate eventually')
    # parser.add_argument('--neg_size_eval_train', type=int, default=500, help='number of negative samples when evaluating training triples')
    
    return parser.parse_args(args)
