# mMARCO
A multilingual version of MS MARCO passage ranking dataset

This repository presents a neural machine translation-based method for translating [MS MARCO passage ranking dataset](https://microsoft.github.io/msmarco/).
The code available here is the same used in our paper **mMARCO: A Multilingual Version of MS MARCO Passage RankingDataset**.

## Dataset
We translate MS MARCO passage ranking dataset, a large-scale IR dataset comprising more than half million anonymized questions that were sampled from Bing's search query logs.

## MarianMNT
To translate the MS MARCO dataset, we use MarianNMT an open-source neural machine translation framework originally written in C++ for fast training and translation. The Language Technology Research Group at the University of Helsinki made available [more than a thousand language pairs](https://huggingface.co/Helsinki-NLP) for translation, supported by HuggingFace framework.

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


## Fine-tuned Models
* [ptT5-base-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-pt-msmarco-100k)
* [ptT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/ptt5-base-en-pt-msmarco-10k)
* [mT5-base-en-pt-msmarco](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco)
* [mT5-base-multi-msmarco](https://huggingface.co/unicamp-dl/mt5-base-multi-msmarco)
* [mMiniLM-pt-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-pt-msmarco)
* [mMiniLM-en-pt-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-en-pt-msmarco)
* [mMiniLM-multi-msmarco](https://huggingface.co/unicamp-dl/multilingual-MiniLM-L6-v2-multi-msmarco)
