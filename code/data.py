import six.moves.cPickle as pickle
#import pandas as pd
import numpy as np
import string
import re
import random
#from keras.preprocessing import sequence
from itertools import *

class input_data():
	def __init__(self, args):
		self.args = args
	
		def load_p_content(path, word_n = 100000):
			f = open(path, 'rb')
			p_content_set = pickle.load(f)
			f.close()

			def remove_unk(x):
				return [[1 if w >= word_n else w for w in sen] for sen in x]

			p_content, p_content_id = p_content_set
			p_content = remove_unk(p_content)

			# padding with max len 
			for i in range(len(p_content)):
				if len(p_content[i]) > self.args.c_len:
					p_content[i] = p_content[i][:self.args.c_len]
				else:
					pad_len = self.args.c_len - len(p_content[i])
					p_content[i] = np.lib.pad(p_content[i], (0, pad_len), 'constant', constant_values=(0,0))
			
			# for i in range(len(p_content)):
			# 	if len(p_content[i]) < self.args.c_len:
			# 		print i
			# 		print p_content[i]
			# 		break

			p_content_set = (p_content, p_content_id)

			return p_content_set

		def load_word_embed(path, word_n = 32784, word_dim = 128):
			word_embed = np.zeros((word_n + 2, word_dim))

			f = open(path,'r')
			for line in islice(f, 1, None):
				index = int(line.split()[0])
				embed = np.array(line.split()[1:])
				word_embed[index] = embed

			return word_embed

		self.p_content, self.p_content_id = load_p_content(path = self.args.datapath + 'content.pkl')
		self.word_embed = load_word_embed(path = self.args.datapath + 'word_embedding.txt')

		self.triple_sample_p = self.compute_sample_p()
		#print self.total_triple_n
	

	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.win_s
		walk_L = self.args.walk_l
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n

		total_triple_n = [0.0] * 9 # nine kinds of triples
		het_walk_f = open(self.args.datapath + "het_random_walk_full.txt", "r")
		centerNode = ''
		neighNode = ''

		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[0] += 1
								elif neighNode[0] == 'p':
									total_triple_n[1] += 1
								elif neighNode[0] == 'v':
									total_triple_n[2] += 1
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[3] += 1
								elif neighNode[0] == 'p':
									total_triple_n[4] += 1
								elif neighNode[0] == 'v':
									total_triple_n[5] += 1
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[6] += 1
								elif neighNode[0] == 'p':
									total_triple_n[7] += 1
								elif neighNode[0] == 'v':
									total_triple_n[8] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / total_triple_n[i]
		print("sampling ratio computing finish.")

		return total_triple_n


	def gen_het_walk_triple_all(self):
		print ("sampling triple relations ...")
		triple_list_all = [[] for k in range(9)] # nine kinds of triples
		window = self.args.win_s
		walk_L = self.args.walk_l
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n
		triple_sample_p = self.triple_sample_p # use sampling to avoid memory explosion

		het_walk_f = open(self.args.datapath + "het_random_walk_full.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, A_n - 1)
									# random negative sampling get similar performance as noise distribution sampling
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[0].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[1].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[2].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[3].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[4].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[5].append(triple)
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[6].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[7].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list_all[8].append(triple)
		het_walk_f.close()

		return triple_list_all



