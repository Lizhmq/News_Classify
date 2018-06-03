'''
corpus_recut的语料没有换行，读入时可能内存溢出
每1M换行
'''

w = open('../corpus/corpus_newline.txt', 'a', encoding='utf-8')
with open('../corpus/corpus_recut.txt', encoding='utf-8') as r:
	while (True):
		block = r.read(1024 * 1024)
		if not block:
			break
		w.write(block+'\r\n')