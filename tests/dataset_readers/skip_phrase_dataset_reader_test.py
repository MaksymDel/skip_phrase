# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase

from skip_phrase.dataset_readers import SkipPhraseDatasetReader


class TestSkipPhraserDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SkipPhraseDatasetReader(window_size=2, pivot_ngram_degree=1)
        dataset = reader.read('tests/fixtures/sentences.txt', )

        instances = []

        instances.append({"pivot_phrase": ["master"], 
                     "context_words": ["of", "all"]})

        instances.append({"pivot_phrase": ["of"],
                     "context_words": ["master", "all", "trades"]})

        instances.append({"pivot_phrase": ["all"],
                     "context_words": ["master", "of", "trades", "jack"]})

        instances.append({"pivot_phrase": ["trades"],
                     "context_words": ["of", "all", "jack", "of"]})

        instances.append({"pivot_phrase": ["jack"],
                     "context_words": ["all", "trades", "of", "none"]})

        instances.append({"pivot_phrase": ["of"],
                     "context_words": ["trades", "jack", "none", "."]})
                
        instances.append({"pivot_phrase": ["none"],
                     "context_words": ["jack", "of", "."]})
        
        instances.append({"pivot_phrase": ["."],
                     "context_words": ["of", "none"]})

        instances.append({"pivot_phrase": ["right"],
                     "context_words": ["away"]})

        instances.append({"pivot_phrase": ["away"],
                      "context_words": ["right"]})

        assert len(dataset.instances) == len(instances)
        
        for i in range(len(instances)):
            fields = dataset.instances[i].fields
            assert [t.text for t in fields["pivot_phrase"].tokens] == instances[i]["pivot_phrase"]
            assert [t.text for t in fields["context_words"].tokens] == instances[i]["context_words"]
