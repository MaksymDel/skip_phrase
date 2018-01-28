from typing import List, Dict
import logging
from overrides import overrides
import tqdm
import nltk

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset import Dataset
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO: separate vocabs? (embeddings) for pivot and context
# Tokenize pivot phrase so that it is one single token ? or move this logic to encder
# Index pivot phrase tokens so that we do w2i for separate words ? oe move this logic to encoder 

@DatasetReader.register("skip_phrase_lines")
class SkipPhraseDatasetReader(DatasetReader):
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data apart from the filepath should be passed to the
    constructor of the ``DatasetReader``.
    """

    def __init__(self,
                 window_size: int = None,
                 pivot_ngram_degree: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        self.window_size = window_size
        self.pivot_ngram_degree = pivot_ngram_degree
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
    @classmethod
    def from_params(cls, params: Params) -> 'SkipPhraseDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        window_size = params.pop('window_size', 5)
        pivot_ngram_degree = params.pop('pivot_ngram_degree', 1)
        params.assert_empty(cls.__name__)

        return cls(window_size = window_size, 
                   pivot_ngram_degree = pivot_ngram_degree, 
                   tokenizer = tokenizer, 
                   token_indexers = token_indexers)
    
    @overrides
    def read(self, file_path: str) -> Dataset:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        instances = []
        data_file = open(file_path, 'r')
        logger.info("Reading instances from lines in file at: %s", file_path)

        for line_num, line in enumerate(tqdm.tqdm(data_file.readlines())):
            if not line:
                continue
            
            sentence = nltk.word_tokenize(line)

            #if len(sentence) == 1:
            #    continue

            sentence_instances = self.sentence_to_instances(sentence)
            instances.extend(sentence_instances)

        if not instances:
            raise ConfigurationError("No instances read!")
        data_file.close()

        return Dataset(instances)

    def sentence_to_instances(self, sentence: List[str]) -> List[Instance]:
        """
        Coverts a sentence to a banch of instances. Senteces should be split by space.
        """
        instances = []
        for i in range(len(sentence)):
            pivot_phrase = [sentence[i]]
            context_words_left = sentence[max(i - self.window_size, 0): i]
            context_words_right = sentence[i + 1: i + 1 + self.window_size]
            instance = self.text_to_instance(" ".join(pivot_phrase), 
                                             " ".join(context_words_left + context_words_right))
            instances.append(instance)
        
        return instances
 
    @overrides
    def text_to_instance(self, pivot_phrase: str, context_words: str = None) -> Instance:
        """
        If context words is None, it means we are just using this for inference to get an 
        embedding vector for pivot phrase

        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        tokenized_pp = self._tokenizer.tokenize(pivot_phrase.lower())
        pp_field = TextField(tokenized_pp, self._token_indexers)
        fields = {'pivot_phrase': pp_field}

        if context_words is not None:
            tokenized_cw = self._tokenizer.tokenize(context_words.lower())
            cw_field = TextField(tokenized_cw, self._token_indexers)
            fields['context_words'] = cw_field

        return Instance(fields)



    