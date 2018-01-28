from typing import List, Tuple

import json

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('skip_phrase_predictor')
class SkipPhrasePredictor(Predictor):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader
    
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        #self._dataset_reader.text_to_instance()
        json_dict = {"pivot_phrase": line}
        return sanitize(json_dict)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        pp = " ".join(outputs["pivot_phrase"])
        emb = "\n".join(str(o) for o in outputs["pivot_phrase_embedding"])
        #print(pp)
        #print(emb)
        res = "\n============================\n" + pp + "\n" + emb + "\n============================\n"
        return res


    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        pivot_phrase = json_dict["pivot_phrase"]
        instance = self._dataset_reader.text_to_instance(pivot_phrase)
        
        pivot_phrase = [t.text for t in instance.fields["pivot_phrase"].tokens]
        tokenized_pivot_phrase = {"pivot_phrase": pivot_phrase}
        return (instance, sanitize(tokenized_pivot_phrase))
