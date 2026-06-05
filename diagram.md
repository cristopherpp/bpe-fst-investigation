+----------------+
|  BPETokenizer  |
+----------------+
| MergeRule      |
| train()        |
| tokenize_word()|
| tokenize()     |
| save_merges()  |
| load_merges()  |
+----------------+
        ^
        |
        | used by
        |
+-------------------+
|  FSTBPETokenizer  |
+-------------------+
| MorphologicalFST  |
| BPETokenizer      |
| train()           |
| tokenize_word()   |
| tokenize()        |
+-------------------+

+--------------------+
|  MorphologicalFST  |
+--------------------+
| State              |
| Transition         |
| MorphologicalRule  |
| add_suffix_rule()  |
| add_root()         |
| segment_word()     |
+--------------------+

+----------------+
|   Benchmark    |
+----------------+
| evaluate()     |
| print_result() |
+----------------+
