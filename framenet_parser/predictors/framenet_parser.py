from typing import List, Dict

import numpy
from overrides import overrides
from spacy.tokens import Doc

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("framenet_parser")
class FramenetParserPredictor(Predictor):

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL model uses a different API for creating instances.")

    def tokens_to_instances(self, tokens):
        words = [token.text for token in tokens]
        lemmas = [token.lemma_.lower() for token in tokens]
        # dummy define
        node_types = []
        node_attrs = []
        origin_lexical_units = []
        p2p_edges = []
        p2r_edges = []
        origin_frames = []
        frame_elements = []
        
        instances = []
        instances.append(self._dataset_reader.text_to_instance(words, lemmas,
                                                               node_types=node_types,
                                                               node_attrs=node_attrs,
                                                               origin_lexical_units=origin_lexical_units,
                                                               p2p_edges=p2p_edges,
                                                               p2r_edges=p2r_edges,
                                                               origin_frames=origin_frames,
                                                               frame_elements=frame_elements))
        return instances

    def _sentence_to_framenet_instances(self, json_dict: JsonDict) -> List[Instance]:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self.tokens_to_instances(tokens)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        outputs = self._model.forward_on_instances(instances)
        results = {"words": instances[0]["metadata"]['sentence'], "frames": outputs[0]["predicted_frames"], "frame_elements": outputs[0]["predicted_roles"]}
        return sanitize(results)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instances = self._sentence_to_framenet_instances(inputs)

        if not instances:
            return sanitize({"verbs": [], "words": self._tokenizer.tokenize(inputs["sentence"])})

        return self.predict_instances(instances)