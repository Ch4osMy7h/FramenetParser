from overrides import overrides

from allennlp.training.metrics.metric import Metric

class RoleMetrics(Metric):
    def __init__(self):
        self.reset()

    @staticmethod
    def merge_neighboring_spans(predictions):
        """
        Merges adjacent spans with the same label, ONLY for the prediction (to encounter spurious ambiguity).
        Returns
        -------
        List[Tuple[int, int, str]]
            where each tuple represents start, end and label of a span.
        """
        merge_predictions = dict()
        target_fe_dict = dict()
        for (span_1, span_2), label in predictions:
            if span_1 not in target_fe_dict:
                target_fe_dict[span_1] = []
            target_fe_dict[span_1].append((span_2[0], span_2[1], label))

        for target, labeled_spans in target_fe_dict.items():
            labeled_spans_set = set(labeled_spans)
            sorted_spans = sorted([x for x in list(labeled_spans_set)])
            prev_start, prev_end, prev_label = sorted_spans[0]
            for span in sorted_spans[1:]:
                if span[2] == prev_label and span[0] == prev_end + 1:
                    # Merge these two spans.
                    labeled_spans_set.remove(span)
                    labeled_spans_set.remove((prev_start, prev_end, prev_label))
                    labeled_spans_set.add((prev_start, span[1], prev_label))
                    prev_end = span[1]
                else:
                    prev_start, prev_end, prev_label = span
            for span in labeled_spans_set:
                merge_predictions[(target, (span[0], span[1]))] = span[2]
        return merge_predictions

    def __call__(self, decoded_roles, metadata_list):
        predicted_roles_dict = decoded_roles["predicted_roles_dict"]
        predicted_merge_roles_list = []
        for predicted_roles, metadata in zip(predicted_roles_dict, metadata_list):
            gold_frame_elements = metadata["frame_elements_dict"]
            self._total_role_gold += len(gold_frame_elements)
            predicted_roles_set = set(predicted_roles.items())
            predicted_merge_roles = self.merge_neighboring_spans(predicted_roles_set)
            predicted_merge_roles_list.append(predicted_merge_roles)

            self._total_role_predicted += len(predicted_merge_roles)
            for (span_1, span_2), label in predicted_merge_roles.items():
                ix = (span_1, span_2)
                if ix in gold_frame_elements and gold_frame_elements[ix] == label:
                    self._total_role_matched += 1.0
                
        decoded_roles["predicted_merge_roles_list"] = predicted_merge_roles_list

    @overrides
    def get_metric(self, reset=False):
        role_recall = self._total_role_matched / (self._total_role_gold + 1e-13)
        role_precision =  self._total_role_matched / (self._total_role_predicted + 1e-13)
        role_f1 = 2.0 * (role_precision * role_recall) / (role_precision + role_recall + 1e-13)

        if reset:
            self.reset()
        all_metrics = {}
        all_metrics["role_precision"] = role_precision
        all_metrics["role_recall"] = role_recall
        all_metrics["role_f1"] = role_f1
        return all_metrics

    @overrides
    def reset(self):
        self._total_role_gold = 0
        self._total_role_matched = 0
        self._total_role_predicted = 0