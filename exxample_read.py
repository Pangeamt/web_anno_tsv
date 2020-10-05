from web_anno_tsv import open_web_anno_tsv

tsv1 = 'test.tsv'

with open_web_anno_tsv(tsv1) as f:
    for i, sentence in enumerate(f):
        print(f"Sentence {i}:", sentence.text)
        for j, annotation in enumerate(sentence.annotations):
            print(f'\tAnnotation {j}:')
            print('\t\tText:', annotation.text)
            print("\t\tLabel:", annotation.label)
            print("\t\tOffsets", f"{annotation.start}, {annotation.stop}")
