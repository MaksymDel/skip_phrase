# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase

from skip_phrase.dataset_readers import SkipPhraseDatasetReader


class TestSkipPhraserDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SkipPhraseDatasetReader(window_size=2, pivot_ngram_degree=1)
        instances_read = reader.read('tests/fixtures/sentences.txt', )

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

        instances.append({"pivot_phrase": ["wrong"]})

        instances.append({"pivot_phrase": ["right"],
                     "context_words": ["away"]})

        instances.append({"pivot_phrase": ["away"],
                      "context_words": ["right"]})

        assert len(instances_read) == len(instances)
        
        #print([i.fields["pivot_phrase"].tokens for i in instances_read])
        #assert False
        
        for i in range(len(instances)):
            fields = instances_read[i].fields
            assert [t.text for t in fields["pivot_phrase"].tokens] == instances[i]["pivot_phrase"]
            try:
                assert [t.text for t in fields["context_words"].tokens] == instances[i]["context_words"]
            except KeyError:
                pass

            
    def test_read_from_file_phrases(self):

        reader = SkipPhraseDatasetReader(window_size=2, pivot_ngram_degree=2)
        instances_read = reader.read('tests/fixtures/sentences-short.txt', )

        instances = []

        instances.append({"pivot_phrase": ["master"], 
                     "context_words": ["of", "all"]})

        instances.append({"pivot_phrase": ["master", "of"], 
                     "context_words": ["all", "trades"]})

        instances.append({"pivot_phrase": ["of"],
                     "context_words": ["master", "all", "trades"]})

        instances.append({"pivot_phrase": ["of", "all"],
                     "context_words": ["master", "trades"]})

        instances.append({"pivot_phrase": ["all"],
                     "context_words": ["master", "of", "trades"]})

        instances.append({"pivot_phrase": ["all", "trades"],
                     "context_words": ["master", "of"]})

        instances.append({"pivot_phrase": ["trades"],
                     "context_words": ["of", "all"]})

        instances.append({"pivot_phrase": ["wrong"]})

        assert len(instances_read) == len(instances)
        
        for i in range(len(instances)):
            fields = instances_read[i].fields
            assert [t.text for t in fields["pivot_phrase"].tokens] == instances[i]["pivot_phrase"]
            try:
                assert [t.text for t in fields["context_words"].tokens] == instances[i]["context_words"]
            except KeyError:
                pass


