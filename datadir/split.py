import os
import pickle

import numpy as np
from sklearn.utils import shuffle

from config import data_config

from filter import get_context_reply



if __name__ == '__main__':
	# it's the proper line, i've commented it to debug, see below
	# context, reply = get_context_reply(data_config['filtered'])

	# now just checking i.e. debugging
	context, reply = get_context_reply(data_config['checking'])


	print('total pairs: {}'.format(len(context)))

	# shuffling data
	# random seed is used for regeneration
	context, reply = shuffle(context, reply, random_state=42)


	segment_count = 0
	data_len = len(context)
	for i in range(0, data_len, data_config['segment_size']):
		end = min(data_len, i + data_config['segment_size'])
		
		s_context = context[i : end]
		s_reply = reply[i : end]

		segment_file_path = data_config['segment_prefix'] + str(segment_count) + '.txt'
		
		with open(segment_file_path, 'w') as segment_file:
			for c, r in zip(s_context, s_reply):
				segment_file.write(c + '\n')
				segment_file.write(r + '\n')

		# finding sentence average to check if each segment would be ok or not
		clen = 0
		rlen = 0
		for c, r in zip(s_context, s_reply):
			clen += len(c)
			rlen += len(r)

		print('segment_{}:'.format(segment_count))
		print('average context len: {:.4f}'.format((1.0 * clen) / (end - i)))
		print('average reply len: {:.4f}'.format((1.0 * rlen) / (end - i)))

		segment_count += 1


	if os.path.isfile(data_config['metadata']):
		with open(data_config['metadata'], 'rb') as metadata_file:
			metadata = pickle.load(metadata_file)
			metadata['segment_n'] = segment_count
		with open(data_config['metadata'], 'wb') as f:
			pickle.dump(metadata, f)
	else:
		print('Error: metadata file {} not found'.format(data_config['metadata']))

