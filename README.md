# Edit+synt: A Syntax-Aware Edit-based System for Text Simplification
This is the official repository of the Paper A Syntax-Aware Edit-based System for Text Simplification presented at the [RANPL 2021 conference](https://ranlp.org/ranlp2021/proceedings-20Sep.pdf).
Please cite this paper if you use our code or system output.
```
@InProceedings{cumbicuspineda-gonzalezdios-soroa:2021:RANLP,
  author    = {Cumbicus-Pineda, Oscar M.  and  Gonzalez-Dios, Itziar  and  Soroa, Aitor},
  title     = {A Syntax-Aware Edit-based System for Text Simplification},
  booktitle      = {Deep Learning for Natural Language Processing Methods and Applications},
  month          = {September},
  year           = {2021},
  address        = {Held Online},
  publisher      = {INCOMA Ltd.},
  pages     = {324--334},
  abstract  = {Edit-based text simplification systems have attained much attention in recent years due to their ability to produce simplification solutions that are interpretable, as well as requiring less training examples compared to traditional seq2seq systems. Edit-based systems learn edit operations at a word level, but it is well known that many of the operations performed when simplifying text are of a syntactic nature. In this paper we propose to add syntactic information into a well known edit-based system. We extend the system with a graph convolutional network module that mimics the dependency structure of the sentence, thus giving the model an explicit representation of syntax. We perform a series of experiments in English, Spanish and Italian, and report improvements of the state of the art in four out of five datasets. Further analysis shows that syntactic information is always beneficial, and suggest that syntax is more helpful in complex sentences.},
  url       = {https://aclanthology.org/2021.ranlp-1.38}
}
```
Our system is based on [EditNTS](https://github.com/yuedongP/EditNTS), but we propose leveraging syntactic informationderived from dependency trees into a well knownedit-based ATS system. We present a syntax awareedit-based system for ATS which uses a graph con-volutional network (GCN) layer to represent the dependency trees. In thetraining process, the GCN learns to refine the rep-resentation of input sentence words according totheir structural relations in the dependency graph.These syntax augmented representations are com-bined with the encoder outputs using a residualconnection, and passed to the decoding stage.
Our experiments confirm the effectiveness of ourapproach, outperforming previous ATS systemsand improving the state-of-the-art results in severaldatasets. 
