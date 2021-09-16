import os
import re
import csv
import sys
import ftfy
import nltk
import torch
import argparse
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader, Dataset

nltk.download('punkt')

class MSMarco(Dataset):
    '''
    Pytorch's dataset abstraction for MSMarco.
    '''

    def __init__(self, file_path, target_language):

        self.documents = self.load_msmarco(file_path)
    def __len__(self):
        return len(self.documents)

    def load_msmarco(self, file_path:str):
        '''
        Returns a list with tuples of [(doc_id, doc)].
        It uses ftfy to decode special carachters.
        Also, the special translation token ''>>target_language<<' is
        added to sentences.

        Args:
            - file_path (str): The path to the MSMarco collection file.
        '''
        documents = []
        with open(file_path, 'r', errors='ignore') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for line in tqdm(csv_reader, desc="Reading .tsv file"):
              doc_id = line[0]
              doc_lines = nltk.tokenize.sent_tokenize(ftfy.ftfy(line[1]))
              for doc in doc_lines:
                if len(doc) > 1:
                    documents.append((doc_id, r'>>{target_language}<< ' + doc))
        
        return documents

    def __getitem__(self,idx):
        doc_id, doc = self.documents[idx]
        return doc_id, doc



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='Helsinki-NLP/opus-mt-en-ROMANCE', type=str, required=False,
                        help="Model to be used on the translation")
    parser.add_argument("--target_language", default=None, type=str, required=True,
                        help="Target language code. The available codes can be found here \
                        https://developers.google.com/admin-sdk/directory/v1/languages")
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help=".tsv file with MSMarco documents to be translated.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Path to save the translated file.")
    parser.add_argument("--batch_size", default=12, type=int, required=False,
                        help="Batch size for translation")
    parser.add_argument("--num_workers", default=4, type=int, required=False,
                        help="Background workers")
    parser.add_argument("--num_beams", default=5, type=int, required=False,
                        help="Beams used in translation decoding")                        
    parser.add_argument("--max_seq_length", default=64, type=int, required=False,
                        help="The maximum total input sequence length after tokenization. Sequences longer \
                            than this will be truncated, sequences shorter will be padded.")
    

    
    args = parser.parse_args()
   
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. \
                Use --overwrite_output_dir to overcome.")
    else:
        os.mkdir(args.output_dir)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MarianTokenizer.from_pretrained(args.model_name_or_path)
    model     = MarianMTModel.from_pretrained(args.model_name_or_path).to(device).eval()

    output_file = args.output_dir + 'translated_' + args.input_file
    train_ds = MSMarco(args.input_file, args.target_language)
    translation_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

    with open(output_file, 'a', encoding='utf-8', errors='ignore') as output:
        for batch in tqdm(translation_loader, desc="Translating..."):
            doc_ids   = batch[0]
            documents = batch[1]
            
            tokenized_text = tokenizer.prepare_seq2seq_batch(
                documents, 
                max_length=args.max_seq_length,
                return_tensors="pt")

            with torch.no_grad():
                translated = model.generate(
                    input_ids=tokenized_text['input_ids'].to(device), 
                    max_length=args.max_seq_length, 
                    num_beams=args.num_beams,
                    do_sample=False)

                translated_documents = tokenizer.batch_decode(
                    translated, 
                    skip_special_tokens=True)
                
            for doc_id, translated_doc in zip(doc_ids, translated_documents):
                output.write(doc_id + '\t' + translated_doc + '\n')
    print("Done!")

if __name__ == '__main__':
    main()
