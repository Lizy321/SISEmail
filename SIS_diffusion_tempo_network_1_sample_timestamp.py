# coding=utf-8
# coding=utf-8
import pandas as pd
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from random import choice
import random
from pandas import DataFrame
import os
import boto3
from botocore.client import Config
import json
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter


def normalize(d):
    d -=  min(d)
    d /= (max(d) - min(d))
    print(max(d),min(d))
    return d

## modified on Mar.6 2019 sample activities within timestep
def convert_timestamp(df_ori,timestep):
    df_new = df_ori[['user1','user2','weight']]
    df_new['timestamp'] = [t-t%timestep for t in df_ori['timestamp']]
    df_new = df_new.drop_duplicates(keep='first')
    df_new.index = range(len(df_new))
    print('len_df_new:',len(df_new))
    # print(df_new.head())
    return df_new


def node_degree_range(degree_norm):

    # 将每条边上的概率变成CDF
    node_list = degree_norm.keys()  # 获取邻居序列，由每个元组组成列表，其中元组形式为（邻居，时间）
    node_Pr_list = degree_norm.values()  # 邻居概率
    node_Pr_list = np.cumsum(node_Pr_list)  # 获取累加概率
    node_Pr_list_cumsum = [0]
    node_Pr_list_cumsum.extend(node_Pr_list)
    node_cPR = {}  # 键为概率范围，值为每个邻居序列（邻居，时间, 权重）
    for nn in range(0, len(node_list)):
        value = node_list[nn]
        key = (node_Pr_list_cumsum[nn], node_Pr_list_cumsum[nn + 1])  # 该邻居处于的概率范围
        node_cPR[key] = value

    return node_cPR

def select_node_by_degree(node_cPR):

    # 随机生成一个概率
    random_p = np.random.rand()

    for key,value in node_cPR.items():
        if key[0]<random_p and key[1] > random_p:
            return value
    return choice(list(node_cPR.values()))


def SI_spreading(args,path,end):


    if args.movie == 1:
        Network = pd.read_csv(path.split('/')[0]+'/u.data', sep='\t', names=['user1', 'user2', 'weight', 'timestamp'])
        # normalize weight, so that the range is between (0,1)
        Network['weight'] = list(np.array(Network['weight']).astype(float)/max(Network['weight']))
    if args.infectious ==1 or args.haggle == 1:
        Network = pd.read_csv(path, sep=',', names=['user1', 'user2', 'weight', 'timestamp'])
        # normalize weight, so that the range is between (0,1)
        Network['weight'] = list(np.array(Network['weight']).astype(float) / max(Network['weight']))
    elif args.reality ==1:
        Network = pd.read_csv(path, header=None,sep='\s+|\t+', engine='python',names=['user1', 'user2', 'weight', 'timestamp'])
    elif args.fb ==1:
        Network = pd.read_csv(path, header=None, sep=',', names=['user1', 'user2', 'timestamp'])
        Network['weight'] = 1  # 对于无权网络，每条边的权重赋值为1
        num_tree = 1350
    elif args.email ==1:
        Network = pd.read_csv(path, header=None, sep=',', names=['user1', 'user2', 'timestamp'])
        Network['weight'] = 1  # 对于无权网络，每条边的权重赋值为1
        num_tree = 180
    else:
        Network = pd.read_csv(path, header=None, sep=',', names=['user1', 'user2', 'timestamp'])
        Network['weight'] = 1  # 对于无权网络，每条边的权重赋值为1

    print("Read file sucessefully!")

    ## modified on Mar.6 2019 sample activities within timestep
    Network = convert_timestamp(Network, args.timestep)

    N = len(set(list(Network['user1']) + list(Network['user2'])))       # 计算网络中节点数N
    x = args.x
    Beta = x * N  # Beta is temporal context window count
    print("Beta is %s" % str(Beta))
    print("N is %s" % str(N))
    print("x is %s" % str(x))

    L = 80 # maximum walk length
    w = 1 # context window size
    num_walk = 10
    beta = args.beta   # 传播概率
    gamma = args.gamma#1/(args.learning_ratio) # Recover rate
    print("gamma is: %f" %gamma)
    # contact_tuple = list(Network.itertuples(index=False, name=None))  # 将网络中的contact表示成tuple， tuple[0], [1]表示contact, tuple[2]表示时间

    #将网络数据按照时间从小到大排序
    Network = Network.sort_values(by=['timestamp'])

    timestamp = sorted(list(set(list(Network['timestamp']))))  # 将timestamp进行排序
    len_ts = len(timestamp)
    # 生成一个df，将边的两端互换
    user1_list = list(Network['user1'])
    user2_list = list(Network['user2'])
    Network1 = pd.DataFrame()
    Network1['user1'] = user2_list   #两列互换
    Network1['user2'] = user1_list
    Network1['timestamp'] = Network['timestamp']
    Network1['weight'] = Network['weight']
    Network_double = pd.concat([Network, Network1])
    # 合并Network1和Network，这样边就是双向的，方便节点查找邻居
    Node = list(set(list(Network['user1']) + list(Network['user2'])))
    Node_appear_time = {}  # 以节点为键，节点出现时间为值
    for key in Node:
        Gama_key = Network_double[Network_double['user1'] == key]
        key_timestamp = list(Gama_key['timestamp'])
        Node_appear_time[key] = key_timestamp
    # max_t = max(timestamp)

    G = nx.Graph()
    edgelist = zip(Network['user1'], Network['user2'])
    G.add_edges_from(edgelist)


    # normalize degree
    degree = dict(nx.degree(G))
    keys = degree.keys()
    values = list(degree.values())
    degree_norm = dict(zip(keys, normalize(np.array(values, dtype=np.float64))))

    dict_node_pro_range = node_degree_range(degree_norm)

    ## 24-01-2019 修改 将degree排序，从degree大的点开始循环
    sorted_node = sorted(degree, key=degree.get,reverse=True)  # sorted nodes based on reversed degree (node with max degree rank the first)

    C = 0
    S_T = []  # all the walks
    e = 0.0
    n_tree = 1

    Spreading_Tree_Dict = {}
    while (Beta - C) > 0:
    # while n_tree < num_tree:

        print ('C is %d' % C)
        n_tree += 1
        # 随机选择一个节点i0
        ## 24-01-2019 修改 将degree排序，从degree大的点开始循环
        try:
            if np.random.rand() < e:
                i0 = select_node_by_degree(dict_node_pro_range)
                print('select node by degree')
            else: i0 = choice(Node)
        except:
            i0 = choice(Node)

        print("==%d is selected==" % i0)

        # 选择开始时间
        t0 = choice(list(Network_double[Network_double['user1'] == i0]['timestamp']))
        while len_ts - timestamp.index(t0) < w:
            i0 = choice(Node)
            print('degree:'+str(degree_norm[i0]*100))
            t0 = choice(Network_double[Network_double['user1'] == i0]['timestamp'])
            # t0 = sorted(Network_double[Network_double['user1'] == i0]['timestamp'])[0]

        Node_spread_tree_i0 = {}
        Node_spread_record_i0 = {}
        I_state = {}  # 记录每个节点的状态
        # initiate status
        for node in Node:
            I_state[node] = 0
        #     Node_spread_record_i0[node] = []
        I_state[i0] = 1

        t1_to_max_list = [t for t in timestamp if t >= t0]  # t0之后的所有时间步
        print(len(t1_to_max_list))
        # I_curve = [(t0, 1)]

        def get_key(dict, value):  # 由字典的值，得到所对应的键的函数
            return [k for k, v in dict.items() if v == value]


        for t in t1_to_max_list:
            Infected_node = get_key(I_state, 1)  # get the infected nodes
            if len(Infected_node)==0:
                print('no more infected node')
                break
            # Node_spread_tree_i0[t] = {}  # assign a empty list for timestep t in order to store the infected edge
            # 找到t时刻的网络
            Network_t = Network[Network['timestamp'] == t]

            try:
                G_t = nx.from_pandas_dataframe(Network_t, 'user1', 'user2', ['weight'])  # t时刻的网络
            except:
                G_t = nx.from_pandas_edgelist(Network_t, 'user1', 'user2', ['weight'])

            # 筛选出在当前时刻网络中出现的感染节点,可直接算Infected_node和G_t.nodes()的交集
            Infected_node_t = list(set(Infected_node).intersection(set(G_t.nodes())))
            if len(Infected_node_t) > 0:  # 如果当前步有感染节点出现
                for In_node in Infected_node_t:
                    # Node_spread_tree_i0[t][In_node] = []
                    nei = list(G_t.neighbors(In_node))
                    Sus_nei = []
                    for neighbors in nei:
                        if I_state[neighbors] == 0:
                            Sus_nei.append(neighbors)
                    for Sus_nei_node in Sus_nei:
                        weight = G_t[In_node][Sus_nei_node]['weight']
                        infection_rate = 1 - ((1 - beta) ** weight)  # 感染概率
                        if np.random.rand() < infection_rate:
                            I_state[Sus_nei_node] = 1
                            # Node_spread_tree_i0[t][In_node].append(Sus_nei_node)
                            if In_node not in Node_spread_record_i0.keys():
                                Node_spread_record_i0[In_node] = []
                            if Sus_nei_node not in Node_spread_record_i0[In_node]:
                                Node_spread_record_i0[In_node].append(Sus_nei_node)

            # 01-28-2019 恢复
            for In_node in Infected_node:
                if np.random.rand() < gamma:
                    I_state[In_node] = 0
                    # print("%d is recovered" %In_node)

        # Spreading_Tree_Dict['start_node'] = i0
        # Spreading_Tree_Dict['spreading_tree_dict'] = Node_spread_record_i0
        # Spreading_Tree_Dict[n_tree+num_tree*args.index] = {}
        # Spreading_Tree_Dict[n_tree+num_tree*args.index][i0] = Node_spread_record_i0

        key_len = 0
        for key, value in Node_spread_record_i0.items():
            if value != {}:
                key_len += len(value)
        if key_len != 0:
            print("num_node in spreading tree]: %f" % key_len)


        L = 20
        w = 2
        nw = 1
        print('num_walk', num_walk * degree_norm[i0])
        while nw < min(num_walk*degree_norm[i0],10):
            walk = []
            walk.append(i0)  # 以传播源i开始的walk
            # 11-02-2019 select walks regardless of time
            if nw >5:
                L -= 0.5
            L=max(L,7)
            while len(walk)<L:
                cur = walk[-1]
                # print(cur)
                if Node_spread_record_i0.has_key(cur):
                    cur_t_nei = Node_spread_record_i0[cur]
                    if len(cur_t_nei) > 0:
                        walk.append(choice(cur_t_nei))  #随机选择一个节点
                    else:break
            nw+=1

            if (len(walk) > w):
                print("walks selected:")
                print(walk)
                S_T.append(walk[:int(L)])
                C = C + (len(walk) - 1 + 1)
            num_walk = num_walk + 1
    #挑选足够数量的walk
    while (C-Beta) > 50:
        rm_walk = random.choice(S_T)
        S_T = filter(lambda v: v not in [rm_walk], S_T)
        C -= len(rm_walk)

    return S_T
    # return Spreading_Tree_Dict

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

    # model.save_word2vec_format(save_path)
    model.wv.save_word2vec_format(save_path)
    return

def select_path(args):
    if (args.music == 1):
        path = 'lastfm-dataset-1k-clean'
        end='.csv'
    if (args.movie == 1):
        path = 'ml-100k'
        end='.csv'
    if (args.email == 1):
        path = 'ia-radoslaw-email'
        end='.edges'
    if(args.fb==1):
        path = 'fb-forum'
        end='.txt'
    if(args.contact==1):
        path = 'ia-contacts_hypertext2009'
        end='.txt'
    if(args.infectious==1):
        path = 'infectious'
        end='.txt'

    if (args.data):
        path = 'data'

    if (args.reality):
        path = 'reality_mining'
        end = '.txt'

    if (args.haggle):
        path = 'haggle'
        end = '.txt'

    if(args.wiki):
        path = 'soc-wiki-elec'
        end='.edges'

    directory = os.path.dirname(path+'/emd/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    return path,end

def main():

    parser = ArgumentParser("SI_temporal",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--x', default=710, type=int,
                        help='x')

    parser.add_argument('--index', default=0, type=int,
                        help='Node index range')

    parser.add_argument('--ratio', default=0.75, type=float,
                        help='Train-test ratio')

    parser.add_argument('--learning_ratio', default=1, type=float,
                        help='learning ratio')

    parser.add_argument('--beta', default=0.1, type=float,
                        help='Train-test ratio')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma')

    parser.add_argument('--timestep', default=1, type=int,
                        help='sample interval')

    parser.add_argument('--reality', default=0, type=int, help='retrieve reality data')

    parser.add_argument('--model',default = 'SI',type=str,
                        help='Select model: SI,SIR')

    parser.add_argument('--fb',default=0,type=int,help='retrieve facebook data')

    parser.add_argument('--contact', default=0, type=int,
                        help='retrieve ia-contacts_hypertext2009 data.')

    parser.add_argument('--infectious', default=0, type=int,
                        help='retrieve infectious data')

    parser.add_argument('--haggle', default=0, type=int,
                        help='retrieve haggle data')

    parser.add_argument('--wiki', default=0, type=int,
                        help='retrieve soc-wiki-elec data.')

    parser.add_argument('--music', default=0, type=int,
                        help='retrieve music data.')

    parser.add_argument('--movie', default=0, type=int,
                        help='retrieve movie data.')

    parser.add_argument('--data',default = 0,type=str)#data/data.txt

    parser.add_argument('--email', default=0, type=int,
                        help='retrieve email data.')

    args = parser.parse_args()

    path,end = select_path(args)

    #### connect to aws s3
	s3 = boto3.resource(
		's3',
		region_name='us-east-1',
		aws_access_key_id=AKIAIESWP6R6MOOVCYVQ,
		aws_secret_access_key=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
	)

    import glob
    for i in range(5):
        emd_path = glob.glob(path + "/emd/SIS_beta0"+str(int(10*args.beta))+"_x"+str(args.x)+"_*"+str(i)+"_gamma"+str(args.gamma)+"_window_"+str(args.timestep)+".emd")
        train_path = glob.glob(path + '/'+ path+'-trainratio-'+str(args.ratio)+'*'+str(i)+'.edgelist')
        data_path = glob.glob(path+"/"+path+"*"+end)

        if (emd_path == []):

            if (train_path == []):
                from network_train_test_evaluation import monopartite_train_test_classifier
                monopartite_train_test_classifier(args.ratio, data_path[0],path)
            S_T = SI_spreading(args,train_path[0],end)

            # import pickle
            # with open(path+"_SIS_walk" + str(args.index) + "_beta0" + str(int(10*args.beta))+"_x"+str(args.x)+"_gamma"+str(args.gamma)+"_window_"+str(args.timestep)+".txt", "wb") as f:
            #     pickle.dump(S_T, f)
            # print("Finish Writing walk")

            # import json
            # with open('walk/'+path + "/SIS/"+path+'_st_'+str(i)+'_'+str(args.index)+"_beta0" + str(int(10*args.beta))+"_x"+str(args.x)+"_gamma"+str(args.gamma)+"_window_"+str(args.timestep)+'.json', 'w') as fp:
            #     json.dump(S_T, fp)
            filename = 'walk/'+path + "/SIS/"+path+'_st_'+str(i)+'_'+str(args.index)+"_beta0" + str(int(10*args.beta))+"_x"+str(args.x)+"_gamma"+str(args.gamma)+"_window_"+str(args.timestep)+'.json'
            s3.Object('sis-scripts-bucket', filename).put(Body=S_T)



if __name__ == "__main__":
    main()

    
'''
python2 SI_diffusion_tempo_network.py --x 50 --ratio 0.75 --email 1 --beta 1

python2 SIS_diffusion_tempo_network_4_sample_timestamp.py --x 50 --ratio 0.75 --fb 1 --beta 1 --gamma 0.01 --timestep 100

'''


