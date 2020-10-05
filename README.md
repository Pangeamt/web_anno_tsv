# Inception Webanno tsv

Read and write webanno tsv 3.2 files 

## Install
```python
 pip install git+https://github.com/Pangeamt/web_anno_tsv
```

## Read a webanno tsv file
```python
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
```

## Write a webanno tsv file
```python
from web_anno_tsv import open_web_anno_tsv

tsv1 = 'test.tsv'
tsv2 = 'test_write.tsv'

with open_web_anno_tsv(tsv1) as f1:
    with open_web_anno_tsv(tsv2, 'w') as f2:
        for sentence in f1:
            f2.write(sentence)

```

