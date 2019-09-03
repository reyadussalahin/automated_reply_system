from datadir.config import data_config

# model_config = {
# 	'name'                   : 'seq2seq',
# 	'batch_size'             : 64,
# 	'epoch'                  : 20,
# 	'embedding_size'         : 1024,
# 	'hidden_unit'            : 512,
# 	'hidden_layer'           : 3,
# 	'decoder_seq_length'     : data_config['max_reply'],
# 	'saved_model_path'       : 'model.npz',
# }


# for testing i.e. debugging purpose
model_config = {
	'name'                   : 'seq2seq',
	'batch_size'             : 8,
	'epoch'                  : 1,
	'embedding_size'         : 128,
	'hidden_unit'            : 64,
	'hidden_layer'           : 2,
	'decoder_seq_length'     : data_config['max_reply'],
	'learning_rate'          : 0.001,
	'saved_model_path'       : 'model.npz',
}
