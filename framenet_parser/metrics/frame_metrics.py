from overrides import overrides

from allennlp.training.metrics.metric import Metric

class FrameMetrics(Metric):
    def __init__(self):
        self.reset()

    def __call__(self, decoded_frames, metadata_list):
        predicted_frames_dict = decoded_frames["predicted_frames_dict"]
        for predicted_frames, metadata in zip(predicted_frames_dict, metadata_list):
            gold_target_spans = set(metadata["origin_frames_dict"].keys())
            pred_target_spans = set(predicted_frames.keys())

            self._total_target_gold += len(gold_target_spans)
            self._total_target_predicted += len(pred_target_spans)
            self._total_target_matched += len(gold_target_spans & pred_target_spans)

            gold_frame_spans = metadata["origin_frames_dict"]
            self._total_frame_gold += len(gold_frame_spans)
            predicted_frames_set = set(predicted_frames.items())
            self._total_frame_predicted += len(predicted_frames_set)
            for target_ixs, label in gold_frame_spans.items():
                if len(target_ixs) > 1: self._total_disc_target_gold += 1.0

            for target_ixs, label in predicted_frames_set:
                if len(target_ixs) > 1: self._total_disc_target_predicted += 1.0
                if target_ixs in gold_frame_spans and gold_frame_spans[target_ixs] == label:
                    self._total_frame_matched += 1
                    if len(target_ixs) > 1: self._total_disc_target_matched += 1.0

    @overrides
    def get_metric(self, reset=False):
        target_recall = self._total_target_matched / (self._total_target_gold + 1e-13)
        target_precision =  self._total_target_matched / (self._total_target_predicted + 1e-13)
        target_f1 = 2.0 * (target_precision * target_recall) / (target_precision + target_recall + 1e-13)

        frame_recall = self._total_frame_matched / (self._total_frame_gold + 1e-13)
        frame_precision =  self._total_frame_matched / (self._total_frame_predicted + 1e-13)
        frame_f1 = 2.0 * (frame_precision * frame_recall) / (frame_precision + frame_recall + 1e-13)

        disc_target_recall = self._total_disc_target_matched / (self._total_disc_target_gold + 1e-13)
        disc_target_precision = self._total_disc_target_matched / (self._total_disc_target_predicted + 1e-13)
        disc_target_f1 = 2.0 * (disc_target_precision * disc_target_recall) / (disc_target_precision + disc_target_recall + 1e-13)

        if reset:
            self.reset()
        all_metrics = {}
        all_metrics["target_precision"] = target_precision
        all_metrics["target_recall"] = target_recall
        all_metrics["target_f1"] = target_f1
        all_metrics["frame_precision"] = frame_precision
        all_metrics["frame_recall"] = frame_recall
        all_metrics["frame_f1"] = frame_f1
        all_metrics["disc_target_precision"] = disc_target_precision
        all_metrics["disc_target_recall"] = disc_target_recall
        all_metrics["disc_target_f1"] = disc_target_f1
        return all_metrics

    @overrides
    def reset(self):
        self._total_target_gold = 0
        self._total_target_matched = 0
        self._total_target_predicted = 0
        self._total_frame_gold = 0
        self._total_frame_matched = 0
        self._total_frame_predicted = 0
        self._total_disc_target_gold = 0
        self._total_disc_target_matched = 0
        self._total_disc_target_predicted = 0