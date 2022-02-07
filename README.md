# mMARCO [<img src="https://img.shields.io/badge/arXiv-2108.13897-b31b1b.svg">](https://arxiv.org/abs/2108.13897)
**mMARCO** is a multilingual version of the MS MARCO passage ranking dataset.
For more information, checkout our paper:
  * [**mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset**](https://arxiv.org/abs/2108.13897)
<!---
This repository presents a neural machine translation-based method for translating the MS MARCO passage ranking dataset.
The code available here is the same used in our paper [**mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset**](https://arxiv.org/abs/2108.13897).
-->

We translate MS MARCO passage ranking dataset, a large-scale IR dataset comprising more than half million anonymized questions that were sampled from Bing's search query logs. **mMARCO** includes 14 languages (including the original English version).

All files, including the translated triples, collection, queries (training and validation) and run files, are available in [:hugs: Datasets](https://huggingface.co/datasets/unicamp-dl/mmarco).

```python
>>> dataset = load_dataset('unicamp-dl/mmarco', 'english')
>>> dataset['train'][1]
{'query': 'what fruit is native to australia', 'positive': 'Passiflora herbertiana. A rare passion fruit native to Australia. (...)'}
```

**The old/deprecated version (v1) of mMARCO is available at [README_old.md](README_old.md)**

## Released Model Checkpoints
Our available fine-tuned models are:


| Model | Description | EN | PT |
| :--- | :--- | :---: | :---: |
|[ptT5-base-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-pt-msmarco-100k-v2)| a [PTT5](https://github.com/unicamp-dl/PTT5) model fine-tuned on Portuguese MS MARCO | 0.200 | 0.299 |
|[ptT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2) | a PTT5 model fine-tuned on English and Portuguese MS MARCO| 0.354 | 0.301 |
|[mT5-base-en-msmarco](https://huggingface.co/unicamp-dl/mt5-base-en-msmarco) |a [mT5](https://github.com/google-research/multilingual-t5) model fine-tuned on English MS MARCO | 0.371| 0.293 |
|[mT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco-v2) |a mT5 model fine-tuned on both English and Portuguese MS MARCO | 0.374 | **0.306** |
|[mT5-base-multi-msmarco](https://huggingface.co/unicamp-dl/mt5-base-mmarco-v2) |a mT5 model fine-tuned on mMARCO |0.366 | 0.302|
|[mMiniLM-en-msmarco](https://huggingface.co/unicamp-dl/mMiniLM-L6-v2-en-msmarco) |a [mMiniLM](https://github.com/microsoft/unilm/tree/master/minilm) model fine-tuned on English MS MARCO | **0.382** | 0.277 |
|[mMiniLM-en-pt-msmarco](https://huggingface.co/unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2) |a mMiniLM model fine-tuned on both English and Portuguese MS MARCO | 0.374 | 0.299|
|[mMiniLM-multi-msmarco](https://huggingface.co/unicamp-dl/mMiniLM-L6-v2-mmarco-v2) |a mMiniLM model fine-tuned on mMARCO | 0.366| 0.277|

EN and PT columns refer to MRR@10 on the dev set of English and Portuguse MS MARCO, respectively.

## How To Translate
In order to allow other users to translate the MS MARCO passage ranking dataset to other languages (or a dataset of your own will), we provide the ```translate.py``` script. This script expects a .tsv file, in which each line follows a ```document_id \t document_text``` format.
```
python translate.py --model_name_or_path Helsinki-NLP/opus-mt-{src}-{tgt} --target_language tgt_code--input_file collection.tsv --output_dir translated_data/
```
After translating, it is necessary to reassemble the file, as the documents were split into sentences.
```
python create_translated_collection.py --input_file translated_data/translated_file --output_file translated_{tgt}_collection
```
Translating the entire passages collection of MS MARCO took about 80 hours using a Tesla V100.

# BM25 Baseline for Portuguese
The steps reported here are the same used for any language from mMARCO. 

## Data Prep

Using [pygaggle](https://github.com/castorini/pygaggle) scripts, we convert the mMARCO Portuguese collection into JSON files:
```
python pygaggle/tools/scripts/msmarco/convert_collection_to_jsonl.py \
    --collection-path path/to/portuguese_collection.tsv \
    --output-folder collections/portuguese-msmarco-passage/collection_jsonl
```
## Indexing using [Pyserini](https://github.com/castorini/pyserini)
Now we can index the Portuguese collection using Pyserini:
```
python -m pyserini.index -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator \
    -threads 1 -input collections/portuguese-msmarco-passage/collection_jsonl/ \
    -index indexes/portuguese-lucene-index-msmarco \
    -storePositions -storeDocvectors -storeRaw -language portuguese
```
As the original English set, the built index should have 8,841,823 documents.

## Retrieval
Using a pygaggle script, we select only the queries that are in the qrels file:
```
python pygaggle/tools/scripts/msmarco/filter_queries.py \
    --qrels path/to/qrels.dev.small.tsv \
    --queries path/to/portuguese_queries.dev.tsv \
    --output collections/portuguese-msmarco-passage/portuguese_queries.dev.small.tsv
```
This script results a file with 6980 queries. Now we can retrieve from our index:
 
  ```
python -m pyserini.search --topics collections/portuguese-msmarco-passage/portuguese_queries.dev.small.tsv \
     --index indexes/portuguese-lucene-index-msmarco \
     --language portuguese \
     --output runs/run.portuguese-msmarco-passage.dev.small.tsv  \
     --bm25 --output-format msmarco --hits 1000 --k1 0.82 --b 0.68
  ```
 ## Evaluation
Using the official MS MARCO evaluation script:
```
python pygaggle/tools/scripts/msmarco/msmarco_passage_eval.py \
    path/to/qrels.dev.small.tsv runs/run.portuguese-msmarco-passage.dev.small.tsv
``` 
The output should be like:
```
#####################
MRR @10: 0.152
QueriesRanked: 6980
#####################
```

## Re-ranking with mT5
Finally, we can re-rank our BM25 initial run using [mT5-base-multi-msmarco](https://huggingface.co/unicamp-dl/mt5-base-multi-msmarco) (or each one of the previous listed models):
``` 
python reranker.py --model_name_or_path=unicamp-dl/mt5-base-en-pt-msmarco-v2 \
    --initial_run runs/run.portuguese-msmarco-passage.dev.small.tsv  \
    --corpus path/to/portuguese_collection.tsv \
    --queries portuguese_queries.dev.small.tsv \
    --output_run runs/run.mt5-reranked-portuguese-msmarco-passage.dev.small.tsv
``` 
Using the official MS MARCO evaluation script to evaluate the re-ranked results:
```
python pygaggle/tools/scripts/msmarco/msmarco_passage_eval.py \
    path/to/qrels.dev.small.tsv runs/run.mt5-reranked-portuguese-msmarco-passage.dev.small.tsv
``` 
The output should be like:
```
#####################
MRR @10: 0.306
QueriesRanked: 6980
#####################
```

## Training mMiniLM
An example of mMiniLM-based models training is provided in `train_minilm.py` script.

```
python train_minilm.py --output_dir ./mminilm-pt --language portuguese
```
 
# How to Cite

If you extend or use this work, please cite the [paper][paper] where it was
introduced:

```
@misc{bonifacio2021mmarco,
      title={mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset}, 
      author={Luiz Henrique Bonifacio and Vitor Jeronymo and Hugo Queiroz Abonizio and Israel Campiotti and Marzieh Fadaee and  and Roberto Lotufo and Rodrigo Nogueira},
      year={2021},
      eprint={2108.13897},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[paper]: https://arxiv.org/abs/2108.13897
