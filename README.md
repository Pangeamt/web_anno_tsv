from ada.web_anno_tsv import open_web_anno_tsv

tsv1 = 'data/difficult (3).tsv'

with open_web_anno_tsv(tsv1) as f:
    for sentence in f:
        print("Sentence", sentence.text+"\n")
        for annotation in sentence.annotations:
            print('Text:', annotation.text)
            print("Label:", annotation.label)
            print("Offsets", f"{annotation.start}, {annotation.stop}")
            print("")
        print("----------------")