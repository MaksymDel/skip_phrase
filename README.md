# skip_phrase
Compositional phrase embeddings

Pytroch implementation of skip-gram model to embed phrases compositionally.

For each phrase, individual phrase's words pass through an encoder (RNN/CNN/etc) to get a phrase embedding. Then we predict surrounding words from this phrase embedding.

Allows getting an embedding for every phrase that consists of known individual words. 

Uses negative sampling and [https://github.com/allenai/allennlp](allennlp)  DL4NLP library.
