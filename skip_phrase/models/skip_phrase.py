import sys, traceback
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
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.modules.token_embedders import Embedding

# TODO: WHENEVER I CREATE VARIABLE (OR MAYBE TENSOR), MAKE SURE IT'S ON THE SAME DEVICE AS EVERYTHING ELSE
@Model.register("skip_phrase")
class SkipPhrase(Model):
    """
    Skip phrase model
    """
    def __init__(self,
                 vocab: Vocabulary,
                 pivot_phrase_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 negative_sampling: bool = True,
                 num_negative_examples: int = 10) -> None:
        super().__init__(vocab, regularizer)

        self.negative_sampling = negative_sampling
        self.num_negative_examples = num_negative_examples
        self.pivot_phrase_embedder = pivot_phrase_embedder
        self.vocab_size = self.vocab.get_vocab_size("words")
        self.encoder = encoder
        self._output_projection_layer = Linear(encoder.get_output_dim(), self.vocab_size)
        self._context_words_embedder = Embedding(self.vocab_size, pivot_phrase_embedder.get_output_dim())
        
        initializer(self)
    
    def _get_loss_negative_sampling(self,
                  logits: torch.LongTensor,
                  context_words_tokens: torch.LongTensor,
                  context_words_mask: torch.LongTensor):        
        loss = util.sequence_cross_entropy_with_logits(logits, context_words_tokens, context_words_mask)
        return loss
    
    def _get_loss_context_words(slef,
                                embedded_context: torch.FloatTensor, 
                                embedded_pivot_phrase: torch.FloatTensor, 
                                context_words_mask: torch.LongTensor,
                                batch_average: bool = True) -> torch.FloatTensor:
        """
        Returns
        -------
        A torch.FloatTensor representing the cross entropy loss.
        If ``batch_average == True``, the returned loss is a scalar.
        If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).

        """
        # (batch_size, num_context_words, emb_size) x (batch_size, emb_size, 1) -> (batch_size, num_context_words, 1)
        loss_context_words = torch.bmm(embedded_context, embedded_pivot_phrase.unsqueeze(1).transpose(1, 2))
        # (batch_size, num_context_words)
        loss_context_words = loss_context_words.squeeze()
        # (batch_size, num_context_words)
        loss_context_words = loss_context_words.sigmoid().log()
        # (batch_size, num_context_words)
        loss_context_words = loss_context_words * context_words_mask.float()
        # (batch_size,); 
        # here we add 1e-13 to omit division by zero; 
        # however numerator is zero anyway due to applying mask above
        per_batch_loss = loss_context_words.sum(1) / (context_words_mask.sum(1).float() + 1e-13)

        # make sure there are no infs, that rarely happens
        # per_batch_loss = per_batch_loss.clamp(min=1e-18, max=1e18)

        if batch_average:
            # (scalar)  
            num_non_empty_sequences = ((context_words_mask.sum(1) > 0).float().sum() + 1e-13)
            # (scalar)
            return per_batch_loss.sum() / num_non_empty_sequences
        
        return per_batch_loss

    
    def _get_loss_negative_examples(self,
                                    embedded_negative_examples: torch.FloatTensor, 
                                    embedded_pivot_phrase: torch.FloatTensor,
                                    num_context_words: int, 
                                    batch_average: bool = True) -> torch.FloatTensor:
        """
        Returns
        -------
        A torch.FloatTensor representing the cross entropy loss.
        If ``batch_average == True``, the returned loss is a scalar.
        If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).

        """
        # (batch_size, num_context_words * num_negative_examples, 1)
        loss_negative_examples = torch.bmm(embedded_negative_examples, embedded_pivot_phrase.unsqueeze(1).transpose(1, 2))
        # (batch_size, num_context_words * num_negative_examples)
        loss_negative_examples = loss_negative_examples.squeeze().sigmoid().log()
        # (batch_size, num_context_words, num_negative_examples)
        loss_negative_examples = loss_negative_examples.view(-1, num_context_words, self.num_negative_examples)
        
        


        # TODO: TO DELELTE - DEBUG
        print("loss neg ex iNF SUM:", numpy.isinf(loss_negative_examples.data).cuda().sum(2).sum(1).sum(0))
        print("loss neg ex iNF:", loss_negative_examples[numpy.isinf(loss_negative_examples.data).cuda()])

        # (batch_size) 
        per_batch_loss = loss_negative_examples.sum(2).mean(1)
        
        # TODO: TO DELELTE - DEBUG
        print("per batch loss iNF SUM:", sum(numpy.isinf(per_batch_loss.data).cuda()))
        print("per batch loss iNF:", per_batch_loss[numpy.isinf(per_batch_loss.data).cuda()])

        per_batch_loss.mean()

        # make sure there are no infs, that rarely happens
        #per_batch_loss = per_batch_loss.clamp(min=1e-18, max=1e18)

        if batch_average:
            # (scalar)
            return per_batch_loss.mean()
        
        return per_batch_loss

    def _sample_negative_examples(self,
                                  batch_size: int,
                                  num_context_words: int) -> Dict[str, torch.LongTensor]:
        # defines how many we need
        negative_examples = torch.LongTensor(batch_size, num_context_words * self.num_negative_examples)
        # samples random indexes from vocab uniformly
        if torch.cuda.device_count() > 0: # if there is a GPU
            return torch.autograd.Variable(negative_examples.random_(0, self.vocab_size - 1)).cuda()
        else: # CPU
            return torch.autograd.Variable(negative_examples.random_(0, self.vocab_size - 1))

    def forward(self,
                pivot_phrase: Dict[str, torch.Tensor],
                context_words: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # (batch_size, num_pivot_phrase_tokens, embedding_size)
        embedded_pivot_phrase_tokens = self.pivot_phrase_embedder(pivot_phrase)
        # (batch_size, num_pivot_phrase_tokens)
        pivot_phrase_tokens_mask = util.get_text_field_mask(pivot_phrase)
        # (batch_size, embedding_size)
        embedded_pivot_phrase = self.encoder(embedded_pivot_phrase_tokens, pivot_phrase_tokens_mask)
        
        output_dict = {"pivot_phrase_embedding": embedded_pivot_phrase}

        if context_words is not None:
            # (batch_size, num_context_words)
            context_words_tokens = context_words["words"]            
            # (batch_size, num_context_words)
            context_words_mask = util.get_text_field_mask(context_words)

            loss = None

            if self.negative_sampling: # Negative sampling
                # Compute loss for context words
                # (batch_size, num_context_words, embedding_size)
                embedded_context = self._context_words_embedder(context_words_tokens)
                # (batch_size)
                loss_context_words = self._get_loss_context_words(embedded_context, 
                                                                embedded_pivot_phrase, 
                                                                context_words_mask,
                                                                False)
                
                # Compute loss for negative examples
                batch_size, num_context_words = context_words_tokens.size()
                # (batch_size, num_context_words * num_negative_examples) 
                negative_examples = self._sample_negative_examples(batch_size, num_context_words)
                # (batch_size, num_context_words * num_negative_examples, embedding_size)
                embedded_negative_examples = self._context_words_embedder(negative_examples).neg()
                # (batch_size)
                loss_negative_examples = self._get_loss_negative_examples(embedded_negative_examples, 
                                                                        embedded_pivot_phrase, 
                                                                        num_context_words,
                                                                        False)

                # Compute overall loss
 #               try:
                #loss = -(loss_context_words + loss_negative_examples).mean()
                # debug
                loss1 = loss_context_words.mean()
                loss2 = loss_negative_examples.mean()
                loss_sum = loss1 + loss2
                loss = - loss_sum

#                except:
#                    traceback.print_exc(file=sys.stdout)
#                    print("loss_context_words", loss_context_words)
#                    print("loss_negative_examples", loss_negative_examples)
#                    print("CONTINUE\n\n\n\n\n\n\n")

            else: # Naive version that computes softmax over whole voacb 
                # (batch_size, num_context_words, vocab_size)
                num_context_words = context_words_tokens.size()[1]
                # (batch_size, vocab_size)
                logits = self._output_projection_layer(embedded_pivot_phrase)            
                # (batch_size, num_context_words, vocab_size)
                logits = logits.unsqueeze(1).expand(logits.size()[0], num_context_words, logits.size()[1]).contiguous()
                # (scalar)            
                loss = self._get_loss_negative_sampling(logits, context_words_tokens, context_words_mask)

            output_dict["loss"] = loss

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ToxicModel':
        embedder_params = params.pop("pivot_phrase_embedder")
        pivot_phrase_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2VecEncoder.from_params(params.pop("encoder"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        negative_sampling = params.pop("negative_sampling", True)
        num_negative_examples = params.pop("num_negative_examples", 10)
        # check if there are unprocessed parameters
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   pivot_phrase_embedder=pivot_phrase_embedder,
                   encoder=encoder,
                   initializer=initializer,
                   regularizer=regularizer,
                   negative_sampling=negative_sampling,
                   num_negative_examples=num_negative_examples)
