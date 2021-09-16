import argparse
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help=".tsv file with MSMarco documents to be translated.")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help=".tsv file with MSMarco documents to be translated.")

    args = parser.parse_args()

    with open(args.input_file, 'r') as f_in:
        with open(args.output_file, 'w') as f_out:
            current_id = -1
            for line in tqdm(f_in):

                doc_id, document = line.split('\t')
                if current_id == -1:
                    current_id = doc_id

                if doc_id != current_id:
                    f_out.write(current_id + '\t' + current_document)
                    current_id = doc_id
                    current_document = document

                else:
                    current_document = current_document + ' ' + document

            # Write tha last document
            f_out.write(current_id + '\t' + current_document)

    print("Done!")

if __name__ == '__main__':
    main()