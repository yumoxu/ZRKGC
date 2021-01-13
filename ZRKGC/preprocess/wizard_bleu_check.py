import os
import sys
import json
from wizard_generator import data_generator
from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)

from cmu_dog_bleu_check import (move_stop_words, normalize_answer, bleu_metric, bleu_stats)

data_path = sys.argv[1]

text_truncate=128
max_knowledge=10000
knowledge_truncate=32
label_truncate=32
max_query_turn=4

def truncate(str, num):
	str = str.strip()
	length = len(str.split())
	list = str.split()[max(0, length - num):]
	return " ".join(list)

def detokenize(tk_str):
	tk_list = tk_str.strip().split()
	r_list = []
	for tk in tk_list:
		if tk.startswith('##') and len(r_list) > 0:
			r_list[-1] = r_list[-1] + tk[2:]
		else:
			r_list.append(tk)
	return " ".join(r_list)


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


if __name__ == "__main__":
	bleu_list = calc_bleu(data_path)	
	bleu_stats(bleu_list)
