from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util

@Model.register("skip_phrase")
class SkipPhrase(Model):
    """
    Skip phrase model
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("tokens")
        self.encoder = encoder
        self._output_projection_layer = Linear(encoder.get_output_dim(), self.num_classes)

        initializer(self)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ToxicModel':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2VecEncoder.from_params(params.pop("encoder"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   initializer=initializer,
                   regularizer=regularizer)

    def _get_loss(self,
                  logits: torch.LongTensor,
                  context_words_tokens: torch.LongTensor,
                  context_words_mask: torch.LongTensor):        
        loss = util.sequence_cross_entropy_with_logits(logits, context_words_tokens, context_words_mask)
        return loss

    def forward(self,
                pivot_phrase: Dict[str, torch.Tensor],
                context_words: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        embedded_pivot_phrase = self.text_field_embedder(pivot_phrase)
        pivot_phrase_mask = util.get_text_field_mask(pivot_phrase)
        encoded_pivot_phrase = self.encoder(embedded_pivot_phrase, pivot_phrase_mask)

        output_dict = {"pivot_phrase_embedding": encoded_pivot_phrase}

        if context_words is not None:
            context_words_tokens = context_words["tokens"]
            # (batch_size, num_classes)
            logits = self._output_projection_layer(encoded_pivot_phrase)
            # (batch_size, num_context_words, num_classes)
            num_context_words = context_words_tokens.size()[1]
            logits = logits.unsqueeze(1).expand(logits.size()[0], num_context_words, logits.size()[1]).contiguous()
            context_words_mask = util.get_text_field_mask(context_words)

            loss = self._get_loss(logits, context_words_tokens, context_words_mask)
            output_dict["logits"] = logits
            output_dict["loss"] = loss

        return output_dict