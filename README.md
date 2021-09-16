# mMARCO
A multilingual version of MS MARCO passage ranking dataset

This repository presents a neural machine translation-based method for translating the [MS MARCO passage ranking dataset](https://microsoft.github.io/msmarco/).
The code available here is the same used in our paper [**mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset**](https://arxiv.org/abs/2108.13897).

## Translated Datasets
As described in our work, we made available 8 translated versions of MS MARCO passage ranking dataset.
The translated passages collection and the queries set (training and validation) are available at:
* [Spanish](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/spanish)
* [French](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/french)
* [Portuguese](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/portuguese)
* [Italian](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/italian)
* [Indonesian](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/indonesian)
* [German](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/german)
* [Russian](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/russian)
* [Chinese](https://console.cloud.google.com/storage/browser/msmarco-translated/multi_msmarco/chinese)


## Released Model Checkpoints
Our available fine-tuned models are: 

| Model | Description | MRR@10*|
| :--- | :--- | :---: |
|[ptT5-base-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-pt-msmarco-100k)| a [PTT5](https://github.com/unicamp-dl/PTT5) model fine-tuned on Portuguese MS MARCO | 0.188 |
|[ptT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-en-pt-msmarco-10k) | a PTT5 model fine-tuned on English and Portuguese MS MARCO| 0.343|
|[mT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco) |a [mT5](https://github.com/google-research/multilingual-t5) model fine-tuned on both English and Portuguese MS MARCO | 0.375|
|[mT5-base-multi-msmarco](https://huggingface.co/unicamp-dl/mt5-base-multi-msmarco) |a mT5 model fine-tuned on mMARCO |0.366 |
|[mMiniLM-pt-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-pt-msmarco) |a [mMiniLM](https://github.com/microsoft/unilm/tree/master/minilm) model fine-tuned on Portuguese MS MARCO | - |
|[mMiniLM-en-pt-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-en-pt-msmarco) |a mMiniLM model fine-tuned on both English and Portuguese MS MARCO | 0.375|
|[mMiniLM-multi-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-multi-msmarco) |a mMiniLM model fine-tuned on mMARCO | 0.363|

\* MRR@10 on English MS MARCO

## Dataset
We translate MS MARCO passage ranking dataset, a large-scale IR dataset comprising more than half million anonymized questions that were sampled from Bing's search query logs.

## Translation Model
To translate the MS MARCO dataset, we use MarianNMT an open-source neural machine translation framework originally written in C++ for fast training and translation. The Language Technology Research Group at the University of Helsinki made available [more than a thousand language pairs](https://huggingface.co/Helsinki-NLP) for translation, supported by HuggingFace framework.

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

# BM25 Baseline for Portuguese Language
The steps reported here are the same used for any language from mMARCO. 

## Data Prep

Using [pygaggle](https://github.com/castorini/pygaggle) scripts, we convert the mMARCO Portuguese collection into json files:
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
 --index indexes/portuguese-lucene-index-msmarco --language portuguese \
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
MRR @10: 0.14122873743575773
QueriesRanked: 6980
#####################
```

## Re-ranking with mT5
Finally, we can re-rank our BM25 initial run using [mT5-base-multi-msmarco](https://huggingface.co/unicamp-dl/mt5-base-multi-msmarco) (or each one of the previous listed models):
``` 
python reranker.py --model_name_or_path=unicamp-dl/ptt5-base-en-pt-msmarco-10k \
--initial_run runs/run.portuguese-msmarco-passage.dev.small.tsv  \
--corpus path/to/portuguese_collection.tsv --queries portuguese_queries.dev.small.tsv \
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
MRR @10: 0.2832968344931086
QueriesRanked: 6980
#####################
```
 
# How to Cite

If you extend or use this work, please cite the [paper][paper] where it was
introduced:

```
@misc{bonifacio2021mmarco,
      title={mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset}, 
      author={Luiz Henrique Bonifacio and Israel Campiotti and Roberto Lotufo and Rodrigo Nogueira},
      year={2021},
      eprint={2108.13897},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[paper]: https://arxiv.org/abs/2108.13897
