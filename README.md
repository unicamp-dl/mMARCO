# mMARCO
A multilingual version of MS MARCO passage ranking dataset

This repository presents a neural machine translation-based method for translating the [MS MARCO passage ranking dataset](https://microsoft.github.io/msmarco/).
The code available here is the same used in our paper **mMARCO: A Multilingual Version of MS MARCO Passage RankingDataset**.

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
* [ptT5-base-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-pt-msmarco-100k)
* [ptT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-en-pt-msmarco-10k)
* [mT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco)
* [mT5-base-multi-msmarco](https://huggingface.co/unicamp-dl/mt5-base-multi-msmarco)
* [mMiniLM-pt-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-pt-msmarco)
* [mMiniLM-en-pt-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-en-pt-msmarco)
* [mMiniLM-multi-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-multi-msmarco)
* 
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
# How to Cite

If you extend or use this work, please cite the [paper][paper] where it was
introduced:

```
@article{rosa2021cost,
      title={A cost-benefit analysis of cross-lingual transfer methods},
      author={Rosa, Guilherme Moraes and Bonifacio, Luiz Henrique and de Souza, Leandro Rodrigues and Lotufo, Roberto and Nogueira, Rodrigo},
      journal={arXiv preprint arXiv:2105.06813},
      year={2021}
    }
```

[paper]: https://arxiv.org/abs/2105.06813
