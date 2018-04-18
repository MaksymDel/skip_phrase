# skip_phrase
Compositional phrase embeddings

Pytroch implementation of skip-gram model to embed phrases compositionally.

For each phrase, individual phrase's words pass through an encoder (RNN/CNN/etc) to get a phrase embedding. Then we predict surrounding words from this phrase embedding.

Allows getting an embedding for every phrase that consists of known individual words. 

#### This project essentially reimplements following papers:
* [Exploring phrase-compositionality in skip-gram models](https://arxiv.org/pdf/1607.06208.pdf) (Compositional Phrase Embeddings)
* [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) (Negative Sampling & Subsampling of the frequent words)
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) (Skip-Gram model)

and also allows using RNNs and CNNs for compositional phrase embeddings

Uses negative sampling and [allennlp](https://github.com/allenai/allennlp)  DL4NLP library.
