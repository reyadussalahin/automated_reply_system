import os
import pickle

import numpy as np
from sklearn.utils import shuffle

from config import data_config

from filter import get_context_reply

from vocab import tokenize


def prepare(s_list, max_s_len, word_to_index):
	D = np.zeros((len(s_list), max_s_len), dtype=np.int32)

	unk_count = 0
	total_count = 0
	for i, s in enumerate(s_list):
		for j, w in enumerate(s):
			if w in word_to_index:
				D[i, j] = word_to_index[w]
			else:
				D[i, j] = word_to_index[data_config['unk']]
				unk_count += 1
			total_count += 1

	print('percentage of unk: {}'.format((100.00 * unk_count) / total_count))
	return D


if __name__ == '__main__':
	if not os.path.isfile(data_config['metadata']):
		print('Error: metadata file {} not found'.format(data_config['metadata']))
	else:
		with open(data_config['metadata'], 'rb') as metadata_file:
			metadata = pickle.load(metadata_file)


		# retrieving data from metadata
		word_to_index = metadata['word_to_index']
		index_to_word = metadata['index_to_word']

		segment_n = metadata['segment_n']
		# checking segement_n
		print('segment_n: {}'.format(segment_n))


		for i in range(segment_n):
			segment_file = data_config['segment_prefix'] + str(i) + '.txt'
			segment_dir = data_config['segment_prefix'] + str(i)

			if not os.path.isdir(segment_dir):
				os.mkdir(segment_dir)

			# get context with correspondng reply
			context, reply = get_context_reply(segment_file)
			
			# tokenize context and reply
			context = tokenize(context)
			reply = tokenize(reply)

			# preparing training data
			print('segment_{}:'.format(i))
			context = prepare(context, data_config['max_context'], word_to_index)
			reply = prepare(reply, data_config['max_reply'], word_to_index)

			print('integer representation of context words:')
			for _, c in zip(range(10), context):
				print(c)

			print('')
			print('integer representation of reply words:')
			for _, r in zip(range(10), reply):
				print(r)


			data_len = len(context)
			offset_count = 0
			for j in range(0, data_len, data_config['offset_size']):
				end = min(data_len, j + data_config['offset_size'])

				o_context = context[j : end]
				o_reply = reply[j : end]

				cf_path = os.path.join(segment_dir, data_config['context_prefix'] + str(offset_count) + '.npy')
				rf_path = os.path.join(segment_dir, data_config['reply_prefix'] + str(offset_count) + '.npy')

				np.save(cf_path, o_context)
				np.save(rf_path, o_reply)

				offset_count += 1

			metadata[segment_dir + '_offset_n'] = offset_count

		with open(data_config['metadata'], 'wb') as metadata_file:
			pickle.dump(metadata, metadata_file)
