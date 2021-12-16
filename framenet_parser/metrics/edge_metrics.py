from overrides import overrides
from typing import Tuple, Set

from allennlp.training.metrics.metric import Metric

class EdgeMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, output_edges, metadata_list):
        
        predicted_p2p_edges_list = output_edges["decoded_p2p_edges_dict"]
        predicted_p2r_edges_list = output_edges["decoded_p2r_edges_dict"]

        for predicted_p2p_edges, predicted_p2r_edges, metadata in zip(predicted_p2p_edges_list, 
                                                                                     predicted_p2r_edges_list, 
                                                                                     metadata_list):
            gold_p2p_edges = metadata["p2p_edges_dict"]
            self._total_p2p_edge_gold += len(gold_p2p_edges)
            self._total_p2p_edge_predicted += len(predicted_p2p_edges)
            for (span_1, span_2), label in predicted_p2p_edges.items():
                ix = (span_1, span_2)
                if ix in gold_p2p_edges and gold_p2p_edges[ix] == label:
                    self._total_p2p_edge_matched += 1
            
            gold_p2r_edges = metadata["p2r_edges_dict"]
            self._total_p2r_edge_gold += len(gold_p2r_edges)
            self._total_p2r_edge_predicted += len(predicted_p2r_edges)
            for (span_1, span_2), label in predicted_p2r_edges.items():
                ix = (span_1, span_2)
                if ix in gold_p2r_edges and gold_p2r_edges[ix] == label:
                    self._total_p2r_edge_matched += 1

    @overrides
    def get_metric(self, reset=False):
        p2p_edges_recall = self._total_p2p_edge_matched / (self._total_p2p_edge_gold + 1e-13)
        p2p_edges_precision =  self._total_p2p_edge_matched / (self._total_p2p_edge_predicted + 1e-13)
        p2p_edges_f1 = 2.0 * (p2p_edges_precision * p2p_edges_recall) / (p2p_edges_precision + p2p_edges_recall + 1e-13)

        p2r_edges_recall = self._total_p2r_edge_matched / (self._total_p2r_edge_gold + 1e-13)
        p2r_edges_precision =  self._total_p2r_edge_matched / (self._total_p2r_edge_predicted + 1e-13)
        p2r_edges_f1 = 2.0 * (p2r_edges_precision * p2r_edges_recall) / (p2r_edges_precision + p2r_edges_recall + 1e-13)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()
        all_metrics = dict()
        all_metrics["p2p_edges_precision"] = p2p_edges_precision
        all_metrics["p2p_edges_recall"] = p2p_edges_recall
        all_metrics["p2p_edges_f1"] = p2p_edges_f1
        all_metrics["p2r_edges_precision"] = p2r_edges_precision
        all_metrics["p2r_edges_recall"] = p2r_edges_recall
        all_metrics["p2r_edges_f1"] = p2r_edges_f1
        return all_metrics

    @overrides
    def reset(self):
        self._total_p2p_edge_gold = 0
        self._total_p2p_edge_predicted = 0
        self._total_p2p_edge_matched = 0
        self._total_p2r_edge_gold = 0
        self._total_p2r_edge_predicted = 0
        self._total_p2r_edge_matched = 0