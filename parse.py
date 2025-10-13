import os
import re
import sys
import argparse

# Global Variables
postings_buffer = [] # List of (term, docID, freq) tuples in memory
temp_file_names = [] # List of temporary postings file names
page_table = {} # docID -> passageID
doc_frequency_map = {} # term -> document frequency
total_document_length = 0 # sum of all document lengths
total_documents = 0 # total number of documents

# Constants
DEFAULT_MAX_BUFFER_POSTINGS = 1000000
ASCII_RANGE = set(range(128))

def tokenize_text(text):
    '''
    Tokenize the input text into a list of ASCII tokens.
    Non-ASCII tokens are filtered out / removed.
    '''
    # covert to lowercase and replace punctuation with spaces
    punctuation_re = re.compile(r"[^\w\s]", flags=re.UNICODE)
    removed_punctuation = punctuation_re.sub(" ", text.lower())

    # split on whitespace
    tokens = removed_punctuation.split()

    # filter to ASCII only (helper function)
    def is_token_ascii(token):
        return all(ord(c) < 128 for c in token)
    
    # return tokens that are ASCII only
    return [t for t in tokens if is_token_ascii(t)]

def update_postings_buffer(term_frequency_map, doc_id):
    '''
    Update the global postings buffer and document frequency map with terms from the current document.
    '''
    global postings_buffer
    global doc_frequency_map

    for term, frequency in term_frequency_map.items():

        # Append (term, docID, freq) to postings buffer
        postings_buffer.append((term, doc_id, frequency))

        # doc frequency increments once per (term, doc)
        doc_frequency_map[term] = doc_frequency_map.get(term, 0) + 1

def write_postings_buffer_to_disk(temp_file_index, temp_file_prefix):
    '''
    Write the current postings buffer to a temporary file on disk.
    '''
    global postings_buffer
    global temp_file_names

    # sort postings buffer by (term, docID) not frequency
    postings_buffer.sort(key=lambda p: (p[0], p[1]))

    # ensure directory exists
    os.makedirs(os.path.dirname(temp_file_prefix) or ".", exist_ok=True)

    temp_file_name = f"{temp_file_prefix}{temp_file_index}.txt"
    count = len(postings_buffer)

    # write to file
    with open(temp_file_name, "w", encoding="utf-8") as f:
        for term, docID, freq in postings_buffer:
            f.write(f"{term}\t{docID}\t{freq}\n")

    # record temp file name
    temp_file_names.append(temp_file_name)

    # clear postings buffer
    postings_buffer.clear()

    print(f"[INFO] Wrote {temp_file_name} with {count} postings.")

def save_document_frequencies(doc_freq_file):
    '''
    Save the document frequency map to a file.
    '''
    global doc_frequency_map

    # ensure directory exists
    os.makedirs(os.path.dirname(doc_freq_file) or ".", exist_ok=True)

    # write to file
    with open(doc_freq_file, "w", encoding="utf-8") as f:
        for term, document_frequency in doc_frequency_map.items():
            f.write(f"{term} {document_frequency}\n")

def save_collection_stats(stats_file):
    '''
    Save the total documents and average document length to a file.
    '''
    global total_documents
    global total_document_length
    # ensure directory exists
    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)
    # compute average document length
    avg_len = (total_document_length / total_documents) if total_documents else 0.0

    # write to file
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"{total_documents}\t{avg_len}\n")

def save_page_table(page_table_file):
    '''
    Save the page table (docID to document length mapping) to a file
    '''
    # ensure directory exists
    os.makedirs(os.path.dirname(page_table_file) or ".", exist_ok=True)

    # write to file
    with open(page_table_file, "w", encoding="utf-8") as f:
        for doc_id, length in page_table.items():
            f.write(f"{doc_id} {length}\n")

def parse_data(file_path, temp_file_prefix, max_buffer_postings = DEFAULT_MAX_BUFFER_POSTINGS, outputs_dir = "tmp"):
    '''
    Parse the input TSV file and build temporary postings files and auxiliary statistics files.
    '''
    global total_documents
    global total_document_length
    global postings_buffer
    global page_table

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error opening file at path {file_path}", file=sys.stderr)
        return

    # Ensure output directory exists
    os.makedirs(outputs_dir, exist_ok=True)

    # Initialize variables
    temp_file_index = 0
    # Docuemnts are given in order (0, 1, 2, ...)
    doc_id = 0

    # Read and process the input file line by line
    with open(file_path, "r", encoding="utf-8") as opened_file:
        # Each line is a passage / document
        for line in opened_file:
            # Remove trailing newline
            line = line.rstrip("\n")

            # Skip empty lines
            if not line:
                continue
            
            # Increment total document count
            total_documents += 1

            # Split on first tab into passageID and text
            parts = line.split("\t", maxsplit=1)

            passage_id = parts[0]
            passage_text = parts[1] if len(parts) > 1 else ""

            # Tokenize passage text
            tokens = tokenize_text(passage_text)

            # Get the current document length from the tokens we parsed
            doc_len = len(tokens)
            
            # store the document lengths
            page_table[doc_id] = doc_len

            # Update total document length            
            total_document_length += doc_len
            
            # Build term frequency map for the document
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            # Update postings buffer and document frequency map with term_freq
            update_postings_buffer(term_freq, doc_id)

            # spill if buffer (by count) exceeds threshold
            if len(postings_buffer) >= max_buffer_postings:
                # Update temp file index and write to disk
                temp_file_index += 1

                # write postings buffer and clear it
                write_postings_buffer_to_disk(temp_file_index, temp_file_prefix)

            # Increment document ID
            doc_id += 1

    # spill remaining postings if exiting from while loop
    if postings_buffer:
        temp_file_index += 1
        write_postings_buffer_to_disk(temp_file_index, temp_file_prefix)

    # write to extra files
    save_document_frequencies(os.path.join(outputs_dir, "doc_frequencies.txt"))
    save_collection_stats(os.path.join(outputs_dir, "collection_stats.txt"))
    save_page_table(os.path.join(outputs_dir, "page_table.txt"))
    print("1. Completed parsing stage.")

def argument_parser():
    '''
    Create the command line argument parser.
    '''
    parser = argparse.ArgumentParser(description="Parse a TSV file into temporary files")
    parser.add_argument("input", help="Path to TSV with lines: <passageID>\\t<passageText>")
    parser.add_argument("--prefix", default="tmp/temp_file_", help="Temporary postings file prefix")
    parser.add_argument("--buffer", type=int, default=DEFAULT_MAX_BUFFER_POSTINGS, help="Max postings in RAM before spilling a sorted run")
    parser.add_argument("--outdir", default="out", help="Directory for stats files")
    return parser

if __name__ == "__main__":
    # parse command line arguments
    args = argument_parser().parse_args()

    # Parse the data
    parse_data(args.input, args.prefix, max_buffer_postings=args.buffer, outputs_dir=args.outdir)
