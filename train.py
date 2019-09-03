import os
import pickle
import itertools
import copy
import time
import sys

import numpy as np
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize

from tqdm import tqdm
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask

from datadir.config import data_config

from config import model_config

DATA_ROOT = 'datadir'


def get_offset_len(metadata, segment):
	segment_dir = data_config['segment_prefix'] + str(segment)
	return metadata[segment_dir + '_offset_n']


def load_data(data_root, segment, offset):
    file_dir = os.path.join(data_root, data_config['segment_prefix'] + str(segment))
    context_path = os.path.join(file_dir, data_config['context_prefix'] + str(offset) + '.npy')
    reply_path = os.path.join(file_dir, data_config['reply_prefix'] + str(offset) + '.npy')
    if not os.path.isfile(context_path):
        print('Error: context file {} not found'.format(context_path))
        return None, None
    if not os.path.isfile(reply_path):
        print('Error: reply file {} not found'.format(reply_path))
        return None, None
    context = np.load(context_path)
    reply = np.load(reply_path)
    return context, reply


def load_metadata(data_root):
	metadata_path = os.path.join(data_root, data_config['metadata'])
	if not os.path.isfile(metadata_path):
		print('Error: metadata file {} not found'.format(metadata_path))
		return None

	with open(metadata_path, 'rb') as metadata_file:
		metadata = pickle.load(metadata_file)
	return metadata


def remove_pad_sequences(sequences, pad_id=0):
    sequences_out = copy.deepcopy(sequences)

    for i, _ in enumerate(sequences):
        if sequences[i][-1] != pad_id:
            continue
        for j in range(1, len(sequences[i])):
            if sequences[i][-j-1] != pad_id:
                sequences_out[i] = sequences_out[i][0 : -j]
                break
    return sequences_out


def get_model(vocab_size):
    embedding_size = model_config['embedding_size']
    hidden_unit = model_config['hidden_unit']
    hidden_layer = model_config['hidden_layer']
    decoder_seq_length = model_config['decoder_seq_length']
    model = Seq2seq(
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocab_size, embedding_size=embedding_size),
        n_layer=hidden_layer,
        n_units=hidden_unit,
        cell_enc=tf.keras.layers.GRUCell,
        cell_dec=tf.keras.layers.GRUCell,
        decoder_seq_length=decoder_seq_length,
    )
    return model


def inference(context, word_to_index, index_to_word, top_n):
    unk_id = word_to_index[data_config['unk']]
    start_id = word_to_index[data_config['go']]
    end_word = data_config['eos']
    reply_length = data_config['max_reply']
    # evaluating model
    # model evaluation starts
    model.eval()
    context_id = [word_to_index.get(word, unk_id) for word in word_tokenize(context)]
    reply_id = model(inputs=[[context_id]], seq_length=reply_length, start_token=start_id, top_n=top_n)
    # generating reply
    reply = []
    for index in reply_id[0]:
        word = index_to_word[index]
        if word == end_word:
        	break
        reply = reply + [word]
    return reply


def train(model, optimizer, X_train, y_train, vocab_size, epoch, n_epoch, word_to_index):
	start_id = word_to_index[data_config['go']]
	end_id = word_to_index[data_config['eos']]

	batch_size = model_config['batch_size']
	decoder_seq_length = model_config['decoder_seq_length']

	n_step = len(X_train) // batch_size
	
	loss_count = 0
	iter_count = 0

	# training starts
	# set model in training mode
	model.train()
	for X, y in tqdm(tl.iterate.minibatches(inputs=X_train, targets=y_train, batch_size=batch_size, shuffle=False),
	    total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, n_epoch), leave=False):

	    X = tl.prepro.pad_sequences(X)
	    decoder_input = tl.prepro.sequences_add_start_id(y, start_id=start_id, remove_last=False)
	    decoder_input = tl.prepro.pad_sequences(decoder_input, maxlen=decoder_seq_length)
	    
	    decoder_output = tl.prepro.sequences_add_end_id(y, end_id=end_id)
	    decoder_output = tl.prepro.pad_sequences(decoder_output, maxlen=decoder_seq_length)
	    decoder_output_mask = tl.prepro.sequences_get_mask(decoder_output)

	    with tf.GradientTape() as tape:
	        # get outputs of model
	        output = model(inputs = [X, decoder_input])
	        output = tf.reshape(output, [-1, vocab_size])
	        # computing loss
	        loss = cross_entropy_seq_with_mask(logits=output,
	        	target_seqs=decoder_output,
	        	input_mask=decoder_output_mask)
	        # updating model weights
	        gradient = tape.gradient(loss, model.all_weights)
	        optimizer.apply_gradients(zip(gradient, model.all_weights))
	    
	    loss_count += loss
	    iter_count += 1

	return iter_count, loss_count



if __name__ == '__main__':
	# rtrieving metadata
	metadata = load_metadata(DATA_ROOT)
	word_to_index = metadata['word_to_index']
	index_to_word = metadata['index_to_word']
	pad_id = word_to_index[data_config['pad']]
	unk_id = word_to_index[data_config['unk']]

	# batch and epoch info
	batch_size = model_config['batch_size']
	n_epoch = model_config['epoch']

	print('batch_size: {}\nn_epoch: {}\n'.format(batch_size, n_epoch))

	# print('some word_index:')
	# for _, (i, w) in zip(range(20), enumerate(word_to_index)):
	# 	print('{}: {}'.format(w, i))

	# print('vocab_size: {}'.format(len(word_to_index)))
	

	vocab_size = X_vocab_size = y_vocab_size = len(word_to_index)


	# retrieving segment and offset info from metadata
	segment_n = metadata['segment_n']
	# print('segment_n: {}'.format(segment_n))
	offset_n_list = []
	for i in range(segment_n):
		offset_n = get_offset_len(metadata, i)
		offset_n_list.append(offset_n)
		# print('segment_{}_offset_n: {}'.format(i, offset_n))

	# creating model
	model = get_model(vocab_size)
	if os.path.isfile(model_config['saved_model_path']):
	    load_weights = tl.files.load_npz(name=model_config['saved_model_path'])
	    tl.files.assign_weights(load_weights, model)
	    print('model weights successfully loaded...')


	# # showing model info
	# print('model info:')
	# print(model)

	contexts = [
		'hi', # hi there
		# 'what do you think about cleopatra',
		# 'what are you doing',
		# 'is a horse faster than a snail',
		# 'you surprised me',
		'do you think you are smarter than human being', # i think a human nature would be helping intelligence
	]

	# defining optimizer with learning rate
	optimizer = tf.optimizers.Adam(learning_rate=model_config['learning_rate'])

	segment_status = False
	offset_status = False
	step_status = False

	for epoch in range(n_epoch):
		epoch_iter = 0
		epoch_loss = 0

		start_time = time.time()
		
		print('\nepoch_{}:'.format(epoch + 1))
		
		for segment in range(segment_n):
			segment_iter = 0
			segment_loss = 0

			if segment_status:
				print('\tsegment_{}:'.format(segment))
			
			offset_n = offset_n_list[segment]
			for offset in range(offset_n):
				offset_iter = 0
				offset_loss = 0

				if offset_status:
					print('\t\toffset_{}:'.format(offset))

				X_train, y_train = load_data(DATA_ROOT, segment, offset)
				# X_train = tl.prepro.remove_pad_sequences(X_train.tolist(), pad_id=pad_id)
				# y_train = tl.prepro.remove_pad_sequences(y_train.tolist(), pad_id=pad_id)

				X_train = remove_pad_sequences(X_train.tolist(), pad_id=pad_id)
				y_train = remove_pad_sequences(y_train.tolist(), pad_id=pad_id)
				# print('X_train type: {}'.format(type(X_train)))
				
				# for _, c in zip(range(3), X_train):
				# 	print(c)

				# for _, r in zip(range(3), y_train):
				# 	print(r)


				X_train, y_train = shuffle(X_train, y_train, random_state=42)

				step_count = 0
				for k in range(0, len(X_train), data_config['max_allowed_per_step']):
					if step_status:
						print('\t\t\tstep_{}:'.format(step_count))

					last = min(len(X_train), k + data_config['max_allowed_per_step'])
					X_t = X_train[k : last]
					y_t = y_train[k : last]

					# print('\rsegment_{} | offset_{} | step_{}'.format(segment, offset, step_count))
					step_iter, step_loss = train(model=model,
						optimizer=optimizer,
						X_train=X_t,
						y_train=y_t,
						vocab_size=vocab_size,
						epoch=epoch,
						n_epoch=n_epoch,
						word_to_index=word_to_index)
					if step_status:
						print('\t\t\tstep_loss: {:.4f}'.format(step_loss / step_iter))

					offset_iter += step_iter
					offset_loss += step_loss

					step_count += 1

				if offset_status:
					print('\t\toffset_loss: {:.4f}'.format(offset_loss / offset_iter))

				segment_iter += offset_iter
				segment_loss += offset_loss

			if segment_status:
				print('\tsegment_loss: {:.4f}'.format(segment_loss / segment_iter))

			epoch_iter += segment_iter
			epoch_loss += segment_loss

		end_time = time.time()
		# printing average loss after every epoch
		print('Epoch [{}/{}]: loss {:.6f}\t\ttime {:.4f}s'.format(
			epoch + 1, n_epoch, epoch_loss / epoch_iter, end_time - start_time))

		print('end_id: {}'.format(word_to_index[data_config['eos']]))
		for context in contexts:
		    print("\ncontext: {}".format(context))
		    top_n = 3
		    predict_n = 3
		    for i in range(predict_n):
		        reply = inference(context=context,
		        	word_to_index=word_to_index,
		        	index_to_word=index_to_word,
		        	top_n=top_n)
		        print("reply: {}".format(' '.join(reply)))

		tl.files.save_npz(model.all_weights, name='model.npz')
