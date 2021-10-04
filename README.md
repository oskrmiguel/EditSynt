# Edit+synt: A Syntax-Aware Edit-based System for Text Simplification
This is the official repository of the Paper A Syntax-Aware Edit-based System for Text Simplification presented at the [RANPL 2021 conference](https://ranlp.org/ranlp2021/proceedings-20Sep.pdf).
Please cite this paper if you use our code or system output.
```
@inproceedings{cumbicus2021editsynt,
  title={A Syntax-Aware Edit-based System for Text Simplification},
  author={Cumbicus-Pineda, Oscar M. and Gonzalez-Dios, Itziar and Soroa, Aitor},
  booktitle = {Proceedings of Recent Advances in Natural Language Processing},
  pages = {329--339},
  year={2021}
}
```
Our system is based on [EditNTS](https://github.com/yuedongP/EditNTS), but we propose leveraging syntactic informationderived from dependency trees into a well knownedit-based ATS system. We present a syntax awareedit-based system for ATS which uses a graph con-volutional network (GCN) layer to represent the dependency trees. In thetraining process, the GCN learns to refine the rep-resentation of input sentence words according totheir structural relations in the dependency graph.These syntax augmented representations are com-bined with the encoder outputs using a residualconnection, and passed to the decoding stage.
Our experiments confirm the effectiveness of ourapproach, outperforming previous ATS systemsand improving the state-of-the-art results in severaldatasets. 
