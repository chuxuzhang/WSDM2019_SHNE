import argparse

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data/",
					help='path to store data')
	parser.add_argument('--model_path', type=str, default='../SHNE/',
				   help='path to save model')
	parser.add_argument('--A_n', type = int, default = 28646,
				   help = 'number of author node')
	parser.add_argument('--P_n', type = int, default = 21044,
					help = 'number of paper node')
	parser.add_argument('--V_n', type = int, default = 18,
					help = 'number of venue node')
	parser.add_argument('--embed_d', type = int, default = 128,
					help = 'embedding dimension')
	parser.add_argument('--lr', type = int, default = 0.001,
				   help = 'learning rate')
	parser.add_argument('--mini_batch_s', type = int, default = 200,
				   help = 'mini batch_size')
	parser.add_argument('--batch_s', type = int, default = 20000,
				   help = 'batch_size')
	parser.add_argument('--train_iter_max', type = int, default = 100,
				   help = 'max number of training iteration')
	parser.add_argument('--c_len', type = int, default = 100,
				   help = 'max len of semantic content')
	parser.add_argument('--save_model_freq', type = float, default = 10,
				   help = 'number of iterations to save model')
	parser.add_argument("--train", type=int, default= 1,
					help = 'train/test label')
	parser.add_argument("--random_seed", type=int, default='1',
					help = 'fixed random seed')
	parser.add_argument("--walk_l", type=int, default='30',
					help = 'length of random walk')
	parser.add_argument("--win_s", type=int, default='7',
					help = 'window size for graph context')
	parser.add_argument("--cuda", type=int, default = 0,
					help = 'GPU running label')

	args = parser.parse_args()

	print("------arguments/parameters-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))
	print("---------------------------------")

	return args



