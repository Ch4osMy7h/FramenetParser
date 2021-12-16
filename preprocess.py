# Modified from https://github.com/swabhs/scaffolding/blob/9d77bd3f37/allennlp/data/dataset_readers/framenet/full_text_reader.py

import logging
import codecs
import os
import xml.etree.ElementTree as ElementTree
import json
import numpy as np

from typing import List, Tuple, Dict, Set, Optional

from tqdm import tqdm
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from collections import defaultdict

class FrameNetPreprocess(object):
    def __init__(self):
        self._namespace = {"fn": "http://framenet.icsi.berkeley.edu"}
        self._tokenization_layers = ["BNC", "PENN"]
        self._max_role_width = 15
        # use it for extracting lemmas
        self._spacy_tokenizer = SpacyTokenizer(pos_tags=True, split_on_spaces=True)

    def preprocess(self, path: str, dst_path):
        data_split = path.split("/")[-2]
        json_file = "{}/{}.json".format(dst_path, data_split)

        json_datas = []
        for root, _, directory in list(os.walk(path)):
            for data_file in sorted(directory):
                if not data_file.endswith(".xml"):
                    continue
                data = self.read_single_fulltext_file(os.path.join(root, data_file))
                json_datas.extend(data)

        print("# Path = %s", path)
        print("# Number of instances = %d", len(json_datas))

        with open(json_file, "w") as f:
            for jd in json_datas:
                f.write(json.dumps(jd) + "\n")
    
    def read_single_fulltext_file(self, data_file: str):
        # print(data_file)
        instances = []
        with codecs.open(data_file, "rb", "utf-8") as xml_file:
            tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        full_text_filename = data_file.split("/")[-1]
        is_test_file = "test" in data_file

        for sentence in tqdm(root.findall("fn:sentence", self._namespace)):
            tokens: List[str] = []
            starts: Dict[int, int] = {}
            ends: Dict[int, int] = {}
            predicates: List[List[Tuple[int, int]]] = []
            lexical_units: List[str] = []
            frames: List[str] = []
            frame_elements: List[List[Tuple[int, int, str]]] = []

            sentence_text = sentence.find("fn:text", self._namespace).text
            annotations = sentence.findall("fn:annotationSet", self._namespace)
            for annotation in annotations:
                annotation_id = annotation.attrib["ID"]

                if annotation_id == "2019791":
                    # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
                    continue
                
                if "luName" in annotation.attrib and "frameName" in annotation.attrib:
                    # Ignore the unannotated instances in ONLY dev/train.
                    if annotation.attrib["status"] == "UNANN" and not is_test_file:
                        continue

                    # Get the LU, Frame and FEs for this sentence.
                    lex_unit = annotation.attrib["luName"].split(".")[0]
                    # lex_unit = annotation.attrib["luName"]
                    frame = annotation.attrib["frameName"]
                    if frame == "Test35":  # Bogus frame.
                        continue

                    target_tokens = []
                    frame_element_list = []
                    # Targets and frame-elements.
                    for layer in annotation.findall("fn:layer", self._namespace):
                        layer_type = layer.attrib["name"]
                        if layer_type == "Target":
                            # Recover the target span.
                            target_labels = layer.findall("fn:label", self._namespace)

                            # Some annotations have missing targets - ignore those.
                            if not target_labels:
                                print("Skipping: Missing target label at %s in %s",
                                            annotation.attrib["ID"], full_text_filename)
                                break

                            # There can be discontinous targets.
                            for label in target_labels:
                                try:
                                    start_token = starts[int(label.attrib["start"])]
                                    end_token = ends[int(label.attrib["end"])]
                                except:
                                    print("Skipping: Startd labels missing for target annotation %s in %s",
                                                annotation.attrib["ID"], full_text_filename)
                                    continue
                                target_tokens.append((start_token, end_token))

                        elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                            # Recover the frame elements.
                            for label in layer.findall("fn:label", self._namespace):
                                if "itype" in label.attrib:
                                    continue
                                try:
                                    start_token = starts[int(label.attrib["start"])]
                                    end_token = ends[int(label.attrib["end"])]
                                    frame_element_list.append((start_token, end_token, label.attrib["name"]))
                                except:
                                    print("Skipping: Frame-elements annotated for missing tokenization at annotation %s in %s",
                                                annotation.attrib["ID"], full_text_filename)
                                    continue

                    if not target_tokens:
                        print("Skipping: Missing target in annotation %s in %s", annotation.attrib["ID"], full_text_filename)
                        continue

                    if target_tokens in predicates:
                        print("Skipping: Repeated annotation %s for frame %s in %s", annotation.attrib["ID"], frame, full_text_filename)
                        continue

                    merge_target_tokens = self.merge_spans(target_tokens)
                    merge_target_tokens = list(sorted(merge_target_tokens))
                    predicates.append(merge_target_tokens)

                    lexical_units.append(lex_unit)
                    frames.append(frame)
                    frame_elements.append(frame_element_list)
                else:
                    for layer in annotation.findall("fn:layer", self._namespace):
                        if layer.attrib["name"] not in self._tokenization_layers:
                            continue

                        tokenization = {}
                        for label in layer.findall("fn:label", self._namespace):
                            start = int(label.attrib["start"])
                            end = int(label.attrib["end"])
                            tokenization[(start, end)] = label.attrib["name"]

                        previous_end = -2
                        for start_end in sorted(tokenization):
                            start, end = start_end
                            if start != previous_end + 2:
                                print(
                                    "Fixing: Missing tokenization at annotation %s in %s.", annotation.attrib["ID"], full_text_filename)
                                # Creating a new token.
                                dummy_start = previous_end + 2
                                dummy_end = start - 2
                                raw_tokens = sentence_text[dummy_start: dummy_end + 1]

                                if raw_tokens == "":
                                    continue
                                elif "" in raw_tokens:
                                    raw_tokens = raw_tokens.replace(" ", "")

                                tokens.append(raw_tokens)
                                starts[dummy_start] = len(tokens) - 1
                                ends[dummy_end] = len(tokens) - 1

                            raw_tokens = sentence_text[start: end + 1]
                            if raw_tokens == "":
                                continue
                            elif "" in raw_tokens:
                                raw_tokens = raw_tokens.replace(" ", "")
                            tokens.append(raw_tokens)
                            starts[start] = len(tokens) - 1
                            ends[end] = len(tokens) - 1
                            previous_end = end
                        break
            if not predicates:
                # Sentence with missing target annotations, will be skipped.
                continue

            assert len(predicates) == len(lexical_units) == len(frames) == len(frame_elements)

            # sorted the data by their positions
            index = [i for i in range(len(predicates))]
            tuples = zip(predicates, index)
            sorted_tuples = sorted(tuples, key=lambda x:x[0])

            sorted_index = [tuple_ixs[1] for tuple_ixs in sorted_tuples]
            predicates = [predicates[i] for i in sorted_index]
            lexical_units = [lexical_units[i] for i in sorted_index]
            frames = [frames[i] for i in sorted_index]
            frame_elements = [frame_elements[i] for i in sorted_index]

            instance = self.process_sentence(tokens, predicates, lexical_units, frames, frame_elements)
            
            assert instance["node_types"], "Exist instances without frame-semantic annotations."
            instances.append(instance)
        xml_file.close()

        if not instances:
            print("No instances were read from the given filepath {}. "
                                    "Is the path correct?".format(data_file))
        return instances

    def merge_spans(self, spans: List[Tuple[int, int]]):
        """
        Merges adjacent spans with the same label, ONLY for the prediction (to encounter spurious ambiguity).

        Returns
        -------
        List[Tuple[int, int, str]]
            where each tuple represents start, end and label of a span.
        """
        # Full empty prediction.
        if not spans:
            return spans
        # Create a sorted copy.
        sorted_spans = sorted([x for x in spans])
        prev_start, prev_end = sorted_spans[0]
        for span in sorted_spans[1:]:
            if span[0] == prev_end + 1:
                # Merge these two spans.
                spans.remove(span)
                spans.remove((prev_start, prev_end))
                spans.append((prev_start, span[1]))
                prev_end = span[1]
            else:
                prev_start, prev_end = span
        return list(spans)
        
    def process_sentence(self,
                         tokens: List[str],
                         predicates: List[List[Tuple[int, int]]],
                         lexical_units: List[str],
                         frames: List[str],
                         frame_elements: List[List[Tuple[int, int, str]]]):

        instance = dict()
        instance["sentence"] = tokens

        node_types_dict = dict()
        node_attrs_dict = dict()

        #In this paper, for nodes that are both PPRD and FPRD, we took out the data of evoke different frames.
        # According to our investigation, there is only one in the dataset.
        filter_index_dict = dict()

        # Step 1: Generate predicate node types and their corresponding attributes (i.e. frames)
        # NOTE: We filter the 
        for i, predicate in enumerate(predicates):
            frame = frames[i]
            # single-word predicate
            if len(predicate) == 1:
                sp = predicate[0]
                if sp not in node_attrs_dict:
                    node_attrs_dict[sp] = frame
                assert node_attrs_dict[sp] == frame, "frame conflictions caused by predicates"
                    
                if sp not in node_types_dict:
                    node_types_dict[sp] = "FPRD"
                elif "PPRD" in node_types_dict[sp]:
                    node_types_dict[sp] = "FPRD-PPRD"
            # multi-word predicate
            elif len(predicate) >= 2:
                for sub_word in predicate:
                    if sub_word not in node_attrs_dict:
                        node_attrs_dict[sub_word] = frame
                    elif node_attrs_dict[sub_word] != frame:
                        filter_index_dict[i] = 1
                        break
                    if sub_word not in node_types_dict or node_types_dict[sub_word] == "PPRD":
                        node_types_dict[sub_word] = "PPRD"
                    elif node_types_dict[sub_word] == "FPRD":
                        node_types_dict[sub_word] = "FPRD-PPRD"
                    else:
                        raise Exception("confliction occur in multi-word node generation")
            else:
                raise Exception("targets with multi predicates")
        
        # Step 2: generate role node types
        for i, predicate in enumerate(predicates):
            if i in filter_index_dict:
                continue
            for frame_element in frame_elements[i]:
                span_ix = (frame_element[0], frame_element[1])
                if span_ix not in node_types_dict:
                    node_types_dict[span_ix] = "ROLE"
                elif "ROLE" not in node_types_dict[span_ix]:
                    node_types_dict[span_ix] += "-ROLE"

        # Generate original frame-semantic structures:
        # (1) predicates
        # (2) predicate-specific lexical units 
        # (3) frame tuples: (predicate, frame)
        # (4) frame element triplets: (frame_tuple, (role_start, role_end), role)
        lexical_unit_tuples = []
        frame_tuples = []
        frame_element_triplets = []
        frame_element_triplets_helper_dict = defaultdict(list)

        predicate2predicate_edges = []
        predicate2role_edges = []
        for i, predicate in enumerate(predicates):
            if i in filter_index_dict:
                continue
            lu = lexical_units[i]
            frame = frames[i]
            
            previous_sub_word = None
            for sub_word in predicate:
                if previous_sub_word:
                    predicate2predicate_edges.append([previous_sub_word[0], previous_sub_word[1], sub_word[0], sub_word[1], "Continuous"])
                    predicate2predicate_edges.append([sub_word[0], sub_word[1], previous_sub_word[0], previous_sub_word[1], "Continuous"])
                previous_sub_word = sub_word
            
            frame_tuples.append((predicate, frame))
            lexical_unit_tuples.append((predicate, lu))
            child_frame_element_triplets = []
            for frame_element in frame_elements[i]:
                child_frame_element_triplets.append((frame_tuples[-1], frame_element[0], frame_element[1], frame_element[2]))

            if child_frame_element_triplets is not None:
                child_frame_element_triplets = list(sorted(child_frame_element_triplets))
                frame_element_triplets.extend(child_frame_element_triplets)

            frame_element_triplets_helper_dict[tuple(predicate)] = frame_elements[i]
        
        # limit the maximum span length to 15
        for frame_tuple in frame_tuples:
            target_ixs = tuple(frame_tuple[0])
            frame_elements = frame_element_triplets_helper_dict[target_ixs]
            for fe in frame_elements:
                start, end, label = fe
                diff = end - start + 1
                while diff >= self._max_role_width:
                    for target_ix in target_ixs:
                        predicate2role_edges.append((target_ix[0], target_ix[1], start, start + self._max_role_width - 1, label))
                    start = start + self._max_role_width
                    diff = end - start + 1
                if start <= end:
                    for target_ix in target_ixs:
                        predicate2role_edges.append((target_ix[0], target_ix[1], start, end, label))

        instance["node_types"] = list(node_types_dict.items())
        instance["node_attrs"] = list(node_attrs_dict.items())
        instance["p2p_edges"] = predicate2predicate_edges
        instance['origin_frames'] = frame_tuples
        instance['p2r_edges'] = predicate2role_edges
        instance['frame_elements'] = frame_element_triplets
        instance['origin_lexical_units'] = lexical_unit_tuples

        raw_sentence = " ".join(tokens)
        spacy_tokens = self._spacy_tokenizer.tokenize(raw_sentence)
        assert len(spacy_tokens) == len(tokens)
        lemmas = [sctoken.lemma_.lower() for sctoken in spacy_tokens]
        instance["lemmas"] = lemmas
        return instance

if __name__ == "__main__":
    dst15_path = "data/preprocessed-fn1.5"
    dst17_path = "data/preprocessed-fn1.7"
    os.makedirs(dst15_path, exist_ok=True)
    os.makedirs(dst17_path, exist_ok=True)

    processor = FrameNetPreprocess()
    processor.preprocess("data/fndata-1.5/train/fulltext", dst15_path)
    processor.preprocess("data/fndata-1.5/dev/fulltext", dst15_path)
    processor.preprocess("data/fndata-1.5/test/fulltext", dst15_path)
    
    processor.preprocess("data/fndata-1.7/train/fulltext", dst17_path)
    processor.preprocess("data/fndata-1.7/dev/fulltext", dst17_path)
    processor.preprocess("data/fndata-1.7/test/fulltext", dst17_path)
