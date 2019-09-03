from config import data_config

from nltk.tokenize import word_tokenize


REDDIT = data_config['reddit']
TWITTER = data_config['twitter']
CORNELL = data_config['cornell']

ALL_FILTERED = data_config['filtered']


# returns context with corresponding reply
def get_context_reply(file):
	with open(file, 'r') as f:
		lines = f.read().split('\n')
		lines = [line.strip().lower() for line in lines if len(line) > 0]

		context = []
		reply = []
		for i in range(0, len(lines), 2):
			context.append(lines[i])
			reply.append(lines[i+1])

	return context, reply


def filter(context, reply):
	xc = data_config['max_context']
	nc = data_config['min_context']
	xr = data_config['max_reply']
	nr = data_config['min_reply']
	
	fc = []
	fr = []
	for c, r in zip(context, reply):
		_c = word_tokenize(c)
		_r = word_tokenize(r)
		lc = len(_c)
		lr = len(_r)
		if lc < nc or lc > xc or lr < nr or lr > xr:
			continue
		fc.append(c)
		fr.append(r)
	return fc, fr


if __name__ == '__main__':
	reddit_context, reddit_reply = get_context_reply(REDDIT)
	twitter_context, twitter_reply = get_context_reply(TWITTER)
	cornell_context, cornell_reply = get_context_reply(CORNELL)

	print('> Before filtering:')
	print('len_reddit: {}'.format(len(reddit_context)))
	print('len_twitter: {}'.format(len(twitter_context)))
	print('len_cornell: {}'.format(len(cornell_context)))
	print('len_total: {}'.format(len(reddit_context) + len(twitter_context) + len(cornell_context)))
	
	all_context = reddit_context + twitter_context + cornell_context
	all_reply = reddit_reply + twitter_reply + cornell_reply

	all_context, all_reply = filter(all_context, all_reply)

	print('\n> After filtering:')
	print('len_total: {}'.format(len(all_context)))

	with open(ALL_FILTERED, 'w') as f:
		for c, r in zip(all_context, all_reply):
			f.write(c + '\n')
			f.write(r + '\n')
