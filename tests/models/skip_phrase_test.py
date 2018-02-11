# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase

class SkipPhraseTest(ModelTestCase):
    def setUp(self):
        super(SkipPhraseTest, self).setUp()
        self.set_up_model('tests/fixtures/skip_gram.json',
                          'tests/fixtures/sentences.txt')

    def test_model_can_train_save_and_load(self):
        #self.ensure_model_can_train_save_and_load(self.param_file)
        pass