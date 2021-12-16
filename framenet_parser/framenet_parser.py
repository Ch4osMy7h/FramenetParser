import logging
from typing import Dict, List, Optional, Any
import copy
from allennlp.data.fields.text_field import TextFieldTensors
import numpy as np

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary, Batch
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor, span_extractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from framenet_parser.utils import FrameOntology
from framenet_parser.modules.decoders import NodeBuilder
from framenet_parser.modules.decoders import EdgeBuilder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("framenet_parser")
class FramenetParser(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 modules: Dict[str, Any],
                 feature_size: int,
                 max_span_width: int,
                 loss_weights: Dict[str, float],
                 ontology_path: str,
                 lexical_dropout: float = 0.2,
                 lstm_dropout: float = 0.4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 display_metrics: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(FramenetParser, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
    
        self._loss_weights = loss_weights
        self._display_metrics = display_metrics
        modules = Params(modules)

        span_extractor_input_size = context_layer.get_output_dim()

        self._endpoint_span_extractor = EndpointSpanExtractor(
            span_extractor_input_size,
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=text_field_embedder.get_output_dim()
        )

        self._node_builder = NodeBuilder.from_params(vocab=vocab, params=modules.pop("node"))
        self._edge_builder = EdgeBuilder.from_params(vocab=vocab, params=modules.pop("edge"))

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        if lstm_dropout > 0:
            self._lstm_dropout = torch.nn.Dropout(p=lstm_dropout)
        else:
            self._lstm_dropout = lambda x: x
        
        self._ontology = FrameOntology(ontology_path)
        initializer(self)

    @overrides
    def forward(self,
                text: TextFieldTensors,
                spans: torch.LongTensor,
                node_type_labels: torch.LongTensor = None,
                node_attr_labels: torch.LongTensor = None,
                p2p_edge_labels: torch.IntTensor = None,
                p2r_edge_labels: torch.IntTensor = None,
                node_valid_attrs: torch.LongTensor = None,
                valid_p2r_edges: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # Encoder: BERT + BIHLSTM
        text_embeddings = self._text_field_embedder(text)
        text_embeddings = self._lexical_dropout(text_embeddings)

        # Shape: (batch_size, max_sentence_length)
        text_mask = util.get_text_field_mask(text).float()
        sequence_lengths = util.get_lengths_from_binary_sequence_mask(text_mask)

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._lstm_dropout(self._context_layer(text_embeddings, text_mask))
        assert spans.max() < contextualized_embeddings.shape[1]

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).bool()
        spans = F.relu(spans.float()).long()
    
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        # Decoder: Node Building -> Edge Building
        output_nodes = {'loss': 0}
        output_edges = {'loss': 0}

        if self._loss_weights['node'] > 0:
            output_nodes = self._node_builder(
                spans, span_mask, span_embeddings,
                node_type_labels=node_type_labels, node_attr_labels=node_attr_labels)
        
        if self._loss_weights["edge"] > 0:
            output_edges = self._edge_builder(
                spans, span_mask, span_embeddings, sequence_lengths, output_nodes, 
                self._ontology.simple_lu_frame_map, self._ontology.frame_fe_map, 
                p2p_edge_labels, p2r_edge_labels, metadata)

        output_dict = dict(node=output_nodes, edge=output_edges)
        if self.training:
            loss = (self._loss_weights['node'] * output_nodes['loss'] +
                self._loss_weights['edge'] * output_edges['loss'])

            output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics_node = self._node_builder.get_metrics(reset=reset)
        metrics_edge = self._edge_builder.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_node.keys()) +
                        list(metrics_edge.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
                           list(metrics_node.items()) +
                           list(metrics_edge.items())
                           )

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res

    def forward_on_instances(self, instances):
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            merge_output = self.make_output_human_readable(self(**model_input))
            outputs = merge_output['node']
            outputs.update(merge_output['edge'])

            instance_separated_output = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if output is None:
                    continue

                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif isinstance(output, (int, float)):
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output