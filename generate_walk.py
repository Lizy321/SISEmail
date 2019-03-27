import glob
from gensim.models import Word2Vec
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import json
from random import choice
import networkx as nx
import pandas as pd
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter


def load_data(dataset,i):
    train_path = glob.glob(dataset + '/' + dataset + '-trainratio-' + str(0.75) + '*'+str(i)+'.edgelist')
    Network = pd.read_csv(train_path[0], header=None, sep=',', names=['user1', 'user2', 'timestamp'])
    return Network

def learn_embeddings(S_T, save_path):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # save path is the path where we need to save the embedding
    # save_path = 'emb/karate_SI.emb'
    walks = S_T
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=128, window=1, min_count=0, sg=1, workers=8,
                     iter=1)
    # size is Number of dimensions. Default is 128.
    # window is Context size for optimization. Default is 10.
    # workers is Number of parallel workers. Default is 8.
    # iter is Number of epochs in SGD. Default is 1.

    model.wv.save_word2vec_format(save_path)

    return

def count_list(n_gram):
    count_list_dict = {}
    for walk in n_gram:
        count_list_dict[walk] = count_list_dict.get(walk,0) +1
    return dict(sorted(count_list_dict.items()))

def normalize(d):
    d -=  min(d)
    d /= (max(d) - min(d))
    return d

def retrieve_degree(Network):
    G = nx.Graph()
    edgelist = zip(Network['user1'], Network['user2'])
    G.add_edges_from(edgelist)

    # normalize degree
    degree = dict(nx.degree(G))
    keys = degree.keys()
    values = list(degree.values())
    degree_norm = dict(zip(keys, normalize(np.array(values, dtype=np.float64))))
    return degree_norm

# def plot_w_dist(data,label,dataset):
#     plt.figure()  # you need to first do 'import pylab as plt'
#     plt.grid(True)
#     # plt.bar(data.keys(), data.values(), align='center', alpha=0.5)
#     plt.loglog(data.keys(), data.values(), '.')
#     plt.xlabel('Num of node in Spreadint Tree')
#     plt.ylabel('Frequency')
#     plt.title(dataset+' '+label+ ' Distribution Plot')
#     # plt.xlim([0, len(factorize_uniques)])
#     plt.show()

def retrieve_walk(i0,S_T,sum_node,Node_spread_record_i0,degree_norm):
    L = 20
    w = 1
    num_walk = 10
    nw = 0
    # print('num_walk', num_walk * degree_norm[i0])
    while nw < num_walk*degree_norm[i0]:
        walk = []
        walk.append(i0)
        # 11-02-2019 select walks regardless of time
        L=max(L,7)
        c = 0
        while len(walk)<L:
            cur = str(walk[-1])
            if Node_spread_record_i0.has_key(cur):
                cur_t_nei = Node_spread_record_i0[cur]
                if len(cur_t_nei) > 0:
                    walk.append(choice(cur_t_nei))
                else:break
            else: break


        if (len(walk) > w):
            # print("walks selected:")
            # print(walk)
            S_T.append(walk[:int(L)])
            sum_node = sum_node + (len(walk) - 1 + 1)
        nw += 1


    return S_T,sum_node



def main():


    learning_method = ['_singlewalk', '']
    method = learning_method[1]

    parser = ArgumentParser("SI_temporal",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--x', default='50', type=str,
                        help='x')
    parser.add_argument('--dataset', default='ml-100k', type=str,
                        help='dataset')

    parser.add_argument('--beta', default=10, type=int,
                        help='beta')

    parser.add_argument('--ratio', default=0.75, type=float,
                        help='Train-test ratio')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma')

    parser.add_argument('--step', default=1, type=int,
                        help='time window')

    args = parser.parse_args()


    model = 'SIS'
    dataset = args.dataset
    beta = '0' + str(args.beta)
    print("#####beta: %s" % beta)
    gamma = args.gamma

    gamma = round(gamma, 4)
    print("gamma: %s" % str(gamma))

    for i in range(5):
        Network = load_data(args.dataset,i)
        N = len(set(list(Network['user1']) + list(Network['user2'])))
        LEN_WALK = N * int(args.x)
        # if i == 0:
        #     walk_path = glob.glob(
        #         'walk/' + dataset + '/' + model + '/*st_?_beta' + beta + '*_gamma' + str(gamma) + '_window_' + str(
        #             args.step) + '.json')
        # else:
        walk_path = glob.glob('walk/' + dataset + '/' + model + '/*st_'+str(i)+'_?_beta' + beta + '*_gamma' + str(gamma) + '_window_' + str(args.step) + '.json')
        
        print("num of walk files: %d" % len(walk_path))
        S_T = []
        sum_num_node = 0

        df = pd.DataFrame([])
        key_len_list = []

        while sum_num_node < LEN_WALK:
            path = random.choice(walk_path)
            print(path)

            sp_tree_all = json.loads(open(path).read())
            # d = ast.literal_eval(sp_tree_all)

            for key, value in sp_tree_all.items():
                # print('idx',key)
                for key_s, value_s in value.items():
                    # print('node',key_s)
                    S_T, sum_num_node = retrieve_walk(int(key_s), S_T, sum_num_node, value_s, retrieve_degree(Network))
                    # S_T, sum_num_node = retrieve_walk(key_s, S_T, sum_num_node, value_s, retrieve_degree(Network))

            while (sum_num_node - LEN_WALK) > 50:
                rm_walk = choice(S_T)
                S_T = filter(lambda v: v not in [rm_walk], S_T)
                sum_num_node -= len(rm_walk)

        import pickle
        with open('walk/'+dataset + "/SIS/"+dataset+"_SIS_walk"+str(i)+"_beta" + beta + "_x" + args.x + "_gamma" + str(gamma) + "_window_" + str(args.step) + '.txt','wb') as f:
            pickle.dump(S_T, f)
        print("Finish Writing walk")

        emd_path_new = dataset + '/emd/SIS_beta' + beta + "_x" + args.x + \
                       '_SIS_diffussion_' + dataset + 'trainratio' + str(args.ratio) + 'Tempo_train_set_'+str(i)+'gamma' + str(gamma) + method + '_window_' + str(args.step) + '.emd'

        learn_embeddings(S_T, emd_path_new)


if __name__ == "__main__":
    main()

'''
python2 generate_walk.py --x 200 --dataset fb-forum --ratio 0.75 --beta 5 --gamma 0.01 --step 10 --step 100
    
'''

# dataset = 'ia-contacts_hypertext2009'
# dataset = 'ia-radoslaw-email'
# # dataset = 'ml-100k'
# dataset = 'fb-forum'
# # dataset = 'infectious'
# # dataset = 'haggle'