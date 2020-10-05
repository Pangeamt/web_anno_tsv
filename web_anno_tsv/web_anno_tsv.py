from dataclasses import dataclass, field
from typing import List, Dict, Union, Tuple, Generator, Optional, TextIO
from itertools import groupby
from functools import reduce
import uuid
import re
import os
from collections import defaultdict
from copy import deepcopy
import traceback


class ReadException(Exception):
    pass


@dataclass
class Annotation:
    label: str
    text: str
    start: int
    stop: int
    id: Optional[str] = None


@dataclass
class Span:
    text: str
    start: int
    stop: int
    is_token: bool = False
    id: Optional[str] = None


@dataclass
class SpanAnnotation:
    span: Span
    annotations: Dict[str, str] = field(default_factory={})


@dataclass
class AnnotatedSentence:
    text: str
    tokens: []
    annotations: List[Annotation]


def escape_text(text: str) -> str:
    """
    The Inception doc is wrong about what it is really escaped
    :param text:
    :return:
    """
    text = text.replace('\\', '\\\\')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    return text


def un_escape_text(text: str) -> str:
    """
    The Inception doc is wrong about what it is really escaped
    :param text:
    :return:
    """
    text = re.sub(r'(?<=\\)\\r', '\r', text)
    text = re.sub(r'(?<=\\)\\t', '\t', text)
    return text


class IndexMapper:
    """
    To deal with special Inception string offsets. Cf Appendix B: WebAnno TSV 3.2 File format
    """
    def __init__(self, text):
        self.map: List[Tuple[int, int]] = []
        self.inverse: List[Tuple[int, int]] = []

        for i, char in enumerate(text):
            char_java_length = len(char.encode('utf-16-le')) // 2  # le (little indian) to avoid BOM mark
            if i == 0:
                start = 0
            else:
                start = self.map[-1][1]
            stop = start + char_java_length
            self.map.append((start, stop))

            for _ in range(char_java_length):
                self.inverse.append((i, i+1))

    @staticmethod
    def utf16_blocks(text: str):
        return len(text.encode('utf-16-le')) // 2

    def true_offsets(self, start: int, stop: int) -> Tuple[int, int]:
        return self.inverse[start][0], self.inverse[stop-1][1]

    def java_offsets(self, start: int, stop: int) -> Tuple[int, int]:
        return self.map[start][0], self.map[stop-1][1]


class Reader:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.header: str = ''
        with open(self.file_path, encoding='utf-8') as f:
            header = []
            for line in f:
                line = line.strip()
                if line == '':
                    self.header_size = len(header)
                    self.header = "\n".join(header)
                    break
                header.append(line)

        self.f: Optional[TextIO] = None

    def read(self) -> Generator[AnnotatedSentence, None, None]:
        if not self.f:
            self.open()

        sentence_index: int = 0
        lines: List[str] = []
        n: int = 0

        while True:
            line = self.f.readline()
            n += 1
            if n > self.header_size + 2:
                if line == '\n' or line == "":
                    # A sentence block has stop
                    annotated_sentence = Reader.get_annotated_sentence(lines, sentence_index)
                    yield annotated_sentence
                    lines = []
                    sentence_index += 1
                else:
                    # Remove \n at the stop of the line.
                    # We don't use strip() to preserve other whitespaces if any
                    lines.append(line[0:-1])

            # End of file
            if line == '':
                break

        self.close()

    @staticmethod
    def sentence_part(line) -> Optional[str]:
        """
        :param line:
        :return:    "text" -> for text block
                    token_index -> for tokens and associated sub-tokens (token and sub-tokens are grouped)
        """
        if line.startswith('#Text='):
            return 'text'
        elif line == '':
            return None
        else:
            return re.sub(r"\.[0-9]+", '', line[line.index('-')+1: line.index('\t')])

    @staticmethod
    def get_annotated_sentence(lines: List[str], sentence_index: int) -> AnnotatedSentence:
        sentence = ''
        try:
            spans: Union[List[Span], Span] = []
            annotations = defaultdict(list)
            labels = {}
            for group_index, (group_id, group) in enumerate(groupby(lines, key=lambda l: Reader.sentence_part(l))):
                group = list(group)
                if group_id == 'text':
                    # Extract sentence
                    sentence = "\n".join([un_escape_text(line[6:]) for line in group])

                else:
                    # Extract annotations
                    for j, line in enumerate(group):
                        if line != '':
                            # # Read the token line
                            span_annotation = Reader.read_token_line(line)
                            span = span_annotation.span

                            # Register span
                            spans.append(span)

                            # Annotations
                            for label_id, label in span_annotation.annotations.items():
                                if annotations[label_id] and span.start == annotations[label_id][-1].start:
                                    # In Inception, for some strange reasons,
                                    # The entire token inherits label from the the first sub-word,
                                    # if this sub-word is at the beginning of the token.
                                    # We remove it.
                                    annotations[label_id] = annotations[label_id][:-1]
                                labels[label_id] = label
                                annotations[label_id].append(span)

            # 3 things:
            # - Offsets correction [Cf Appstopix B: WebAnno TSV 3.2 File format]
            # - Make offsets relative to sentence
            # - Validation of the offsets calculation
            mapper: IndexMapper = IndexMapper(sentence)
            first_span_start: Optional[int] = None
            for i, span in enumerate(spans):
                if i == 0:
                    first_span_start = span.start
                span.start, span.stop = mapper.true_offsets(span.start - first_span_start, span.stop - first_span_start)

                error = f"Bad offsets ({span.start}, {span.stop}) for span `{span.text}`"
                assert sentence[span.start: span.stop] == span.text, error

            # Compact annotation
            compacted_annotations = []
            for annotation_id, annotation_parts in annotations.items():
                if len(annotation_parts) > 1:
                    # Check that annotations are compact
                    for p1, p2 in zip(annotation_parts, annotation_parts[1:]):
                        space = sentence[p1.stop: p2.start]
                        error = f"Annotation is not compact between {p1} and {p2}"
                        assert sentence[p1.stop: p2.start].isspace() or not space, error

                    # Compacts
                    start = annotation_parts[0].start
                    stop = annotation_parts[-1].stop
                else:
                    start = annotation_parts[0].start
                    stop = annotation_parts[0].stop

                compacted_annotations.append(Annotation(
                    label=labels[annotation_id],
                    text=sentence[start: stop],
                    start=start,
                    stop=stop))

            # Extract tokens
            tokens = [span for span in spans if span.is_token]

            # Sort annotations
            compacted_annotations.sort(key=lambda a: (a.start, -a.stop, a.label))

            return AnnotatedSentence(sentence, tokens, compacted_annotations)
        except Exception as e:
            tb = traceback.format_exc()
            tb_str = str(tb)
            message = f'Sentence {sentence_index}: `{sentence}` ' + tb_str
            raise ReadException(message)

    @staticmethod
    def read_token_line(line):
        columns = line.split('\t')

        # Id
        span_id = columns[0]

        # Offsets
        start, stop = map(int, columns[1].split('-'))

        # Span text
        span_text = un_escape_text(columns[2])

        # Annotations
        annotations = {}
        if columns[4] != '_':
            for part in columns[4].split('|'):
                res = re.search(r'([^[]*)(\[(\d*)\])*$', part)
                label = res.group(1)
                label_id = res.group(3)
                if not label_id:
                    label_id = str(uuid.uuid4())
                annotations[label_id] = label

        return SpanAnnotation(
            span=Span(
                id=span_id,
                start=start,
                stop=stop,
                text=span_text,
                is_token="." not in span_id
            ),
            annotations=annotations
        )

    def open(self):
        self.f = open(self.file_path, encoding='utf-8')

    def close(self):
        if self.f:
            self.f.close()
            self.f = None

    def __iter__(self):
        return self.read()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Writer:
    DEFAULT_HEADER = "#FORMAT=WebAnno TSV 3.2\n" \
        "#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value"
    
    def __init__(self, file_path: str, overwrite=False):
        self.file_path = file_path
        self.overwrite = overwrite
        self.f: Optional[TextIO] = None
        self.sentence_index = 0
        self.annotation_id = 0
        self.stop_last_sentence = 0
        self.header_written = False

    def __enter__(self):
        if os.path.isfile(self.file_path):
            if self.overwrite:
                os.remove(self.file_path)

        self.f = open(self.file_path, 'w', encoding='utf-8')
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def write_header(self, header):
        self.f.write(header.strip() + "\n" * 2)
        self.header_written = True

    def write(self, sentence: AnnotatedSentence):
        self.sentence_index += 1

        # Write header
        if self.sentence_index == 1 and not self.header_written:
            self.write_header(self.DEFAULT_HEADER)

        # Avoid original sentence alteration
        sentence = deepcopy(sentence)

        # Write text
        self.f.write("\n".join(list(map(lambda s: "\n#Text=" + s, sentence.text.split('\n'), )))+"\n")

        # Add ID to annotations
        for annotation in sentence.annotations:
            annotation.id = str(self.annotation_id)
            self.annotation_id += 1

        # Span annotations
        span_annotations = []

        for token_index, token in enumerate(sentence.tokens):
            # Create token span
            token_span = Span(
                text=token.text,
                start=token.start,
                stop=token.stop,
                is_token=True
            )

            # Annotations
            has_token_annotation = False
            for annotation in sorted(sentence.annotations, key=lambda a: (a.start, a.stop)):
                intersection = slice_intersection(
                    token.start,
                    token.stop,
                    annotation.start,
                    annotation.stop)
                if intersection:
                    start, stop = intersection
                    if start == token.start and stop == token.stop:
                        # Token
                        has_token_annotation = True
                        span_annotations.append(SpanAnnotation(
                            token_span, {annotation.id: annotation.label}
                        ))
                    else:
                        # Sub-token
                        span = Span(
                            text=sentence.text[start: stop],
                            start=start,
                            stop=stop,
                            is_token=False)
                        span_annotations.append(SpanAnnotation(
                            span, {annotation.id: annotation.label}
                        ))

            if not has_token_annotation:
                span_annotations.append(SpanAnnotation(token_span,  {}))

        token_index = 0
        sub_token_index = 0
        for _, group in groupby(sorted(
                span_annotations,
                key=lambda a: (a.span.start, -a.span.stop)), key=lambda a: (a.span.start, a.span.stop)):
            group = list(group)
            first = group[0]

            # Span id
            if first.span.is_token:
                token_index += 1
                sub_token_index = 0
                span_id = f"{self.sentence_index}-{token_index}"
            else:
                sub_token_index += 1
                span_id = f"{self.sentence_index}-{token_index}.{sub_token_index}"

            # Offsets
            mapper: IndexMapper = IndexMapper(sentence.text)
            start, stop = mapper.java_offsets(first.span.start, first.span.stop)
            offsets = f"{start + self.stop_last_sentence}" \
                      f"-{stop + self.stop_last_sentence}"

            # text
            text = escape_text(first.span.text)

            # Annotations
            annotations = reduce(lambda x, y: {**x, **y}, [a.annotations for a in group], {})
            schema = "|".join([f"*[{label_id}]" for label_id, _ in annotations.items()]) \
                if annotations else "_"
            labels = "|".join([f"{label}[{label_id}]" for label_id, label in annotations.items()]) \
                if annotations else "_"

            # Write line
            line = "\t".join([span_id, offsets, text, schema, labels]) + "\n"
            self.f.write(line)

        self.stop_last_sentence += IndexMapper.utf16_blocks(sentence.text) + 1


def slice_intersection(start1, stop1, start2, stop2):
    if start2 >= stop1 or start1 >= stop2:
        return None
    else:
        return max(start1, start2), min(stop1, stop2)


def open_web_anno_tsv(file_path: str, mode: str = 'r', overwrite=False):
    if mode == 'r':
        return Reader(file_path)
    elif mode == 'w':
        return Writer(file_path, overwrite=overwrite)
    else:
        raise ValueError(f'Invalid mode {mode}')