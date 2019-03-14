import random
import numpy as np
from args import read_args
import torch
import os
from torch import optim
from torch.autograd import Variable
from utility import *
from data import input_data
torch.set_num_threads(2)

class Model_Run(object):
	def __init__(self, args):
		super(Model_Run, self).__init__()
		for k, v in vars(args).items(): setattr(self, k, v)

		self.args = args
		self.data_generator = input_data(args)

		p_content = self.data_generator.p_content
		word_embed = self.data_generator.word_embed

		self.model = SHNE_Encoder(args, p_content, word_embed)

		# run with GPU
		if self.args.cuda:
			self.model.cuda()

		# setting optimizer
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=0.0)
		#self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[100], gamma=0.5)


	def train_model(self):
		print ('start training...')
		mini_batch_s = self.args.mini_batch_s
		embed_d = self.args.embed_d
		for i in range(self.args.train_iter_max):
			print ('iteration ' + str(i) + ' ...')
			triple_list_all = self.data_generator.gen_het_walk_triple_all()

			min_len = 1e10
			for ii in range(len(triple_list_all)):
				if len(triple_list_all[ii]) < min_len:
					min_len = len(triple_list_all[ii])
			batch_n = int(min_len / mini_batch_s)

			for k in range(batch_n):
				c_out = torch.zeros([len(triple_list_all), mini_batch_s, embed_d])
				p_out = torch.zeros([len(triple_list_all), mini_batch_s, embed_d])
				n_out = torch.zeros([len(triple_list_all), mini_batch_s, embed_d])

				for triple_index in range(len(triple_list_all)):
					triple_list_temp = triple_list_all[triple_index]
					triple_list_batch = triple_list_temp[k * mini_batch_s : (k + 1) * mini_batch_s]
					c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)

					c_out[triple_index] = c_out_temp
					p_out[triple_index] = p_out_temp
					n_out[triple_index] = n_out_temp

				loss = cross_entropy_loss(c_out, p_out, n_out, embed_d)

				self.optim.zero_grad()
				loss.backward()

				self.optim.step() 

				if k % 100 == 0:
					print ("loss: " + str(loss))

			if i % self.args.save_model_freq == 0:
				torch.save(self.model.state_dict(), self.args.datapath + "SHNE.pt")
				# save embeddings for evaluation
				triple_index = 9 
				a_out, p_out, v_out = self.model([], triple_index)
			print ('iteration ' + str(i) + ' finish.')


	def test_model(self):
		self.model.load_state_dict(torch.load(self.args.datapath  + "SHNE.pt"))
		self.model.eval()

		triple_index = 9 
		# generate embeddings for evaluation
		a_out, p_out, v_out = self.model([], triple_index)


if __name__ == '__main__':
	args = read_args()

	# fix random seeds
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	# model execution 
	model_run = Model_Run(args)

	# train/test model
	if args.train:
		model_run.train_model() # embeddings update
	else:
		model_run.test_model() # save embeddings for different applications



