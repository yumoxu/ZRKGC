import sys
import random
import re
from metrics import bleu_metric
import numpy as np
import nltk

from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

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

src_path = sys.argv[1]
tgt_path = sys.argv[2]
knl_path = sys.argv[3]

with open(src_path, encoding="utf-8") as file:
	SRC = file.readlines()
with open(tgt_path, encoding="utf-8") as file:
	TGT = file.readlines()
with open(knl_path, encoding="utf-8") as file:
	KNL = file.readlines()

def calc_bleu():
	mean_len_knows = 0
	bleu_list = []
	for i in range(len(SRC)):
		query_line = SRC[i].strip().replace(" &lt; SEP &gt; ", "<#Q#>").replace("&apos;", "'")
		tgt_line = TGT[i].strip().replace("&apos;", "'")
		# choice no.3
		knows = nltk.sent_tokenize(
			KNL[i].strip().split(" &lt; SEP &gt; ")[2].replace("&apos;", "'")) + nltk.sent_tokenize(
			KNL[i].strip().split(" &lt; SEP &gt; ")[0].replace("&apos;", "'")) + nltk.sent_tokenize(KNL[i].strip().split(" &lt; SEP &gt; ")[1].replace("&apos;", "'"))

		max_b2 = 0
		check_sentence = ""

		for know_line in knows:
			pro_know = normalize_answer(move_stop_words(know_line.strip()))
			pro_response = normalize_answer(move_stop_words(tgt_line.strip()))
			b1, b2, b3 = bleu_metric([pro_know], [pro_response])
			if b2 >= max_b2:
				max_b2 = b2
				check_sentence = know_line
		bleu_list.append(max_b2)

		mean_len_knows += len(knows)
		use_know_list = knows
		if check_sentence in use_know_list:
			index = use_know_list.index(check_sentence)
			use_know_list[0], use_know_list[index] = use_know_list[index], use_know_list[0]
		else:
			use_know_list[0] = check_sentence
		assert use_know_list.index(check_sentence) == 0

		used_know_line = " <#K#> ".join(use_know_list)

		src_line = query_line + " <#Q2K#> " + used_know_line
		if i % 1000 == 0:
			print("have process {} data / {}".format(i, len(SRC)))
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
	bleu_list = calc_bleu()
	bleu_stats(bleu_list)
