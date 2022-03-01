import difflib
import logging
import os
import json
import pickle
from re import X
import xml.etree.ElementTree as ElementTree
from typing import Dict, Iterable, List, Optional, Tuple, Union
from overrides import overrides

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, TextField, SpanField, SequenceLabelField, MetadataField, AdjacencyField, LabelField, IndexField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

from framenet_parser.utils import MissingDict, merge_spans, FrameOntology

logger = logging.getLogger(__name__)

def format_label_fields(node_types, node_attrs, origin_lexical_units, p2e_edges, p2r_edges, origin_frames, frame_elements):

    node_types_dict = MissingDict("O",
        (
            ((span_start, span_end), target)
            for ((span_start, span_end), target) in node_types
        )
    )

    node_attrs_dict =  MissingDict("O",
        (
            ((span_start, span_end), frame)
            for ((span_start, span_end), frame) in node_attrs
        )
    )

    origin_lus_dict = MissingDict("O",
        (
            ((span_ix[0][0], span_ix[-1][1]), lu)
            for span_ix, lu in origin_lexical_units
        )
    )

    origin_frames_dict = MissingDict("O",
        (
            (tuple([tuple(x) for x in span_ix]), frame)
            for span_ix, frame in origin_frames
        )
    )

    p2p_edges_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in p2e_edges:
        p2p_edges_dict_values.append((((span1_start, span1_end), (span2_start, span2_end)), relation))
    p2p_edges_dict = MissingDict("", p2p_edges_dict_values)

    p2r_edges_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in p2r_edges:
        p2r_edges_dict_values.append((((span1_start, span1_end), (span2_start, span2_end)), relation))
    p2r_edges_dict = MissingDict("", p2r_edges_dict_values)

    frame_elements_dict_values = []
    for ((predicate_ixs, frame_label), role_start, role_end, relation) in frame_elements:
        target_ixs_tuple = tuple([tuple(x) for x in predicate_ixs])
        frame_elements_dict_values.append((((target_ixs_tuple, frame_label), (role_start, role_end)), relation))  
    frame_elements_dict = MissingDict("", frame_elements_dict_values)

    return node_types_dict, node_attrs_dict, origin_lus_dict, p2p_edges_dict, p2r_edges_dict, origin_frames_dict, frame_elements_dict
    

@DatasetReader.register('framenet')
class FramenetParserReader(DatasetReader):

    def __init__(self,
                 max_span_width: int,
                 ontology_path: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._ontology = FrameOntology(ontology_path)

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Loop over the documents.
            js = json.loads(line)
            instance = self.text_to_instance(js["sentence"], js["lemmas"], js["node_types"], js["node_attrs"], js["origin_lexical_units"], 
                                             js["p2p_edges"], js["p2r_edges"], js["origin_frames"], js["frame_elements"])
            yield instance

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         lemmas: List[str],
                         node_types: List[Tuple[Tuple[int, int], str]],
                         node_attrs: List[Tuple[Tuple[int, int], str]],
                         origin_lexical_units: List[Tuple[Tuple[Tuple[int, int]], str]],
                         p2p_edges: List[List[Union[int,str]]],
                         p2r_edges: List[Tuple[Tuple[Tuple[int, int], str], int, int, str]],
                         origin_frames: List[Tuple[Tuple[int, int], str]],
                         frame_elements: List[Tuple[Tuple[Tuple[int, int], str], int, int, str]]):
        
        fields = {}
        text_field = TextField([Token(t) for t in tokens], token_indexers=self._token_indexers)
        fields["text"] = text_field
        
        node_types_dict, node_attrs_dict, origin_lus_dict, p2p_edges_dict, p2r_edges_dict, \
            origin_frames_dict, frame_elements_dict = format_label_fields(node_types, node_attrs, origin_lexical_units,
                                                                          p2p_edges, p2r_edges, 
                                                                          origin_frames, frame_elements)
        
        metadata = dict(sentence=tokens,
                        lemmas=lemmas,
                        node_types_dict=node_types_dict,
                        node_attrs_dict=node_attrs_dict,
                        p2p_edges_dict=p2p_edges_dict,
                        p2r_edges_dict=p2r_edges_dict,
                        p2p_edges=p2p_edges,
                        frame_elements=frame_elements,
                        origin_frames_dict=origin_frames_dict,
                        frame_elements_dict=frame_elements_dict)
                        
        metadata_field = MetadataField(metadata)
        fields["metadata"] = metadata_field
        
        # Span-based output fields.
        spans = []
        node_type_labels_list: Optional[List[str]] = []
        node_attr_labels_list: Optional[List[str]] = []
        node_valid_attrs_list: Optional[List[List[int]]] = [] # use for the comprehensive vocabulary
        valid_p2r_edges_list: Optional[List[List[int]]] = []
        for start, end in enumerate_spans(tokens, max_span_width=self._max_span_width):
            span_ix = (start, end)
            node_type_label = node_types_dict[span_ix]
            node_attr_label = node_attrs_dict[span_ix]

            node_type_labels_list.append(node_type_label)
            node_attr_labels_list.append(node_attr_label)

            lexical_unit = origin_lus_dict[span_ix]
            if lexical_unit in self._ontology.lu_frame_map:
                # valid_frames = self._ontology.lu_frame_map[lexical_unit] + ["O"]
                valid_attrs = self._ontology.lu_frame_map[lexical_unit]
            else:
                valid_attrs = ["O"]
            node_valid_attrs_list.append(ListField([LabelField(x, label_namespace='node_attr_labels') for x in valid_attrs]))

            if node_attr_label in self._ontology.frame_fe_map:
                valid_p2r_edge_labels = self._ontology.frame_fe_map[node_attr_label]
                valid_p2r_edges_list.append(ListField([LabelField(x, label_namespace='p2r_edge_labels') for x in valid_p2r_edge_labels]))
            else:
                valid_p2r_edges_list.append(ListField([LabelField(-1, skip_indexing=True)]))

            spans.append(SpanField(start, end, text_field))
        
        span_field = ListField(spans)
        node_type_labels_field = SequenceLabelField(node_type_labels_list, span_field, label_namespace='node_type_labels')
        node_attr_labels_field = SequenceLabelField(node_attr_labels_list, span_field, label_namespace='node_attr_labels')

        node_valid_attrs_field = ListField(node_valid_attrs_list)
        valid_p2r_edges_field = ListField(valid_p2r_edges_list)

        fields["spans"] = span_field
        fields["node_type_labels"] = node_type_labels_field
        fields["node_attr_labels"] = node_attr_labels_field
        fields["node_valid_attrs"] = node_valid_attrs_field
        fields["valid_p2r_edges"] = valid_p2r_edges_field

        n_spans = len(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans)]

        p2p_edge_labels = []
        p2p_edge_indices = []
        p2r_edge_labels = []
        p2r_edge_indices = []
        for i, j in candidate_indices:
            # becasue i index is nested, j is not nested
            span_pair = (span_tuples[i], span_tuples[j])
            p2p_edge_label = p2p_edges_dict[span_pair]
            p2r_edge_label = p2r_edges_dict[span_pair]
            if p2p_edge_label:
                p2p_edge_indices.append((i, j))
                p2p_edge_labels.append(p2p_edge_label)
            if p2r_edge_label:
                p2r_edge_indices.append((i, j))
                p2r_edge_labels.append(p2r_edge_label)

        p2p_edge_label_field = AdjacencyField(
            indices=p2p_edge_indices, sequence_field=span_field, labels=p2p_edge_labels,
            label_namespace="p2p_edge_labels")

        p2r_edge_label_field = AdjacencyField(
            indices=p2r_edge_indices, sequence_field=span_field, labels=p2r_edge_labels,
            label_namespace="p2r_edge_labels")
        
        fields["p2p_edge_labels"] = p2p_edge_label_field
        fields["p2r_edge_labels"] = p2r_edge_label_field
        
        return Instance(fields)
