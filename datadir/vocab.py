import os
import itertools
import pickle

from nltk.tokenize import word_tokenize
from nltk import FreqDist

from config import data_config

from filter import get_context_reply


def tokenize(lines):
	tk_lines = []

	for line in lines:
		tk_lines.append(word_tokenize(line))

	return tk_lines



if __name__ == '__main__':
	# it's the proper line, i've commented it to debug, see below
	# context, reply = get_context_reply(data_config['filtered'])

	# now just checking i.e. debugging
	context, reply = get_context_reply(data_config['checking'])

	# print(len(context), len(reply))

	tk_context = tokenize(context)
	tk_reply = tokenize(reply)


	# preparing freq_dist
	all_lines_tk = tk_context + tk_reply
	freq_dist = FreqDist(itertools.chain(*all_lines_tk))

	with open(data_config['freq_dist'], 'wb') as freq_dist_file:
		pickle.dump(freq_dist, freq_dist_file)


	# creating vocab of given size
	vocab_size = data_config['vocab_size']
	vocab = freq_dist.most_common(vocab_size)


	# length info
	print('len of freq_dist: {}'.format(len(freq_dist)))
	print('len of vocab: {}'.format(len(vocab)))


	# creating converters
	PAD = data_config['pad']
	UNK = data_config['unk']
	GO = data_config['go']
	EOS = data_config['eos']

	print('first 20 of vocab: {}'.format(vocab[0:20]))

	index_to_word = [PAD] + [UNK] + [word[0] for word in vocab] + [GO] + [EOS]
	word_to_index = dict( [(word, index) for index, word in enumerate(index_to_word)] )

	print('word_index:')
	for _, (i, w) in zip(range(20), enumerate(index_to_word)):
		print('{}: {}'.format(w, i))

	metadata = {
		'word_to_index' : word_to_index,
		'index_to_word' : index_to_word,
	}

	with open(data_config['metadata'], 'wb') as f:
		pickle.dump(metadata, f)
