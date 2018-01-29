#!/usr/bin/env python
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from skip_phrase.dataset_readers.reader import *
from skip_phrase.models.skip_phrase import *
from skip_phrase.predictors.predictor import *

from allennlp.commands import main

if __name__ == "__main__":
    main(prog="python run.py",
         predictor_overrides={'skip_phrase': 'skip_phrase_predictor'})