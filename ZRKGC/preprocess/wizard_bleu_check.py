import os
import sys
import json
from wizard_generator import data_generator
from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)

import re
from metrics import bleu_metric
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import numpy as np
from scipy.stats import describe

def move_stop_words(str):
	item = " ".join([w for w in str.split() if not w.lower() in stop_words])
	return item

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""

	def remove_articles(text):
		return re_art.sub(' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		return re_punc.sub(' ', text)  # convert punctuation to spaces

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))

data_path = sys.argv[1]

if "random" in data_path:
	data_type="wizard_random"
elif "topic" in data_path:
	data_type = "wizard_topic"
else:
	print("WRONG DATATYPE")


def calc_bleu(data_file):
	bleu_list = []
	debug = True
	for (history, label, knowledge_sentences) in data_generator(data_file):
		# process knowledge
		checked_sentence = knowledge_sentences[0]
		
		if debug:
			print('checked_sentence: {}, label: {}'.format(checked_sentence, label))
			debug = False

		pro_know = normalize_answer(move_stop_words(checked_sentence.strip()))
		pro_response = normalize_answer(move_stop_words(label.strip()))
		b1, b2, b3 = bleu_metric([pro_know], [pro_response])
		bleu_list.append(b2)
	
	return bleu_list


def bleu_stats(bleu_list):
	ratio_10 = [bleu for bleu in bleu_list if bleu>=0.1]
	ratio_30 = [bleu for bleu in ratio_10 if bleu>=0.3]
	ratio_40 = [bleu for bleu in ratio_30 if bleu>=0.4]
	ratio_50 = [bleu for bleu in ratio_40 if bleu>=0.5]
	ratio_70 = [bleu for bleu in ratio_50 if bleu>=0.7]
	print(describe(np.array(bleu_list)))
	total = float(len(bleu_list))
	print('10+: {:.2f}, 30+: {:.2f}, 40+: {:.2f}, 50+: {:.2f}, 70+: {:.2f}'.format(
		len(ratio_10)*100/total, len(ratio_30)*100/total, len(ratio_40)*100/total, 
		len(ratio_50)*100/total, len(ratio_70)*100/total))


if __name__ == "__main__":
	bleu_list = calc_bleu(data_path)	
	bleu_stats(bleu_list)
