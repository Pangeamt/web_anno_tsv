from web_anno_tsv import open_web_anno_tsv

tsv1 = 'test.tsv'
tsv2 = 'test_write.tsv'

with open_web_anno_tsv(tsv1) as f1:
    with open_web_anno_tsv(tsv2, 'w') as f2:
        for sentence in f1:
            f2.write(sentence)
