import os
import re
import sys
import argparse
import struct

# Global Variables
postings_cache = [] # List of (term, docID, freq) tuples in memory
page_table = {} # docID -> passageID
doc_frequency_map = {} # term -> document frequency
total_document_length = 0 # sum of all document lengths
total_documents = 0 # total number of documents

# Constants
DEFAULT_MAX_CACHE_SIZE = 5000000
OUTPUTS_DIR = "out"

def update_postings_cache(term_frequency_map, doc_id):
    '''
    Update the global postings cache and document frequency map with terms from the current document.
    '''
    global postings_cache
    global doc_frequency_map

    for term, frequency in term_frequency_map.items():

        # Append (term, docID, freq) to postings buffer
        postings_cache.append((term, doc_id, frequency))

        # doc frequency increments once per (term, doc)
        doc_frequency_map[term] = doc_frequency_map.get(term, 0) + 1

def write_postings_cache_to_file_on_disk(current_file_index, temp_file_name):
    '''
    Write the current postings buffer to a temporary file on disk then clear it for next use
    '''
    global postings_cache

    # sort postings cache by (term, docID)
    postings_cache.sort(key=lambda p: (p[0], p[1]))

    # ensure directory exists
    os.makedirs(os.path.dirname(temp_file_name) or ".", exist_ok=True)

    current_file_name = f"{temp_file_name}{current_file_index}.txt"

    # write to file
    with open(current_file_name, "w", encoding="utf-8") as f:
        for term, docID, freq in postings_cache:
            f.write(f"{term}\t{docID}\t{freq}\n")

    # clear postings cache for next use
    postings_cache.clear()

def save_doc_freqs():
    '''
    Save the document frequency map to a file.
    '''
    global doc_frequency_map

    doc_freq_file = os.path.join(OUTPUTS_DIR, "document_frequencies.txt")

    # ensure directory exists
    os.makedirs(os.path.dirname(doc_freq_file) or ".", exist_ok=True)

    # write to file
    with open(doc_freq_file, "w", encoding="utf-8") as f:
        for term, document_frequency in doc_frequency_map.items():
            f.write(f"{term} {document_frequency}\n")

def save_stats():
    '''
    Save the total documents and average document length to a file.
    '''
    global total_documents
    global total_document_length

    stats_file = os.path.join(OUTPUTS_DIR, "stats.txt")

    # ensure directory exists
    os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)

    # compute average document length for stats
    avg_len = (total_document_length / total_documents) if total_documents else 0.0

    # write to file
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"{total_documents}\t{avg_len}\n")

def save_page_table():
    '''
    Save the page table (docID to document length mapping) to a file
    '''
    # ensure directory exists
    page_table_file = os.path.join(OUTPUTS_DIR, "page_table.txt")
    
    os.makedirs(os.path.dirname(page_table_file) or ".", exist_ok=True)

    # write to file
    with open(page_table_file, "w", encoding="utf-8") as f:
        for doc_id, length in page_table.items():
            f.write(f"{doc_id} {length}\n")

def process_document_text(text):
    '''
    Processes the input document text with Non-ASCII removed
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

def parse_data(input_file, temp_file_prefix, max_cache_size, outputs_dir):
    '''
    Parse the input TSV file and build temporary postings files and extra statistics files.
    '''
    
    # global variables
    global total_documents
    global total_document_length
    global postings_cache
    global page_table

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error opening {input_file}", file=sys.stderr)
        return

    # Ensure output directory exists
    os.makedirs(outputs_dir, exist_ok=True)

    # Prepare document store files for writing offsets of passages
    doc_store_path = os.path.join(outputs_dir, "documents.dat")
    doc_offsets_path = os.path.join(outputs_dir, "doc_offsets.txt")

    # Open document store for binary writes and offsets for text writes
    doc_store_out = open(doc_store_path, 'wb')
    doc_offsets_out = open(doc_offsets_path, 'w', encoding='utf-8')

    current_file_index = 0

    doc_id = 0

    # Read and process the input file line by line
    with open(input_file, "r", encoding="utf-8") as opened_file:
        # Each line is document id and passage text
        for line in opened_file:
            # Remove trailing newline
            line = line.rstrip("\n")

            # Skip empty lines
            if not line:
                continue
            
            # Increment total document count
            total_documents += 1

            # Split on first tab into document ID and document text
            parts = line.split("\t", maxsplit=1)

            document_text = parts[1] if len(parts) > 1 else ""

            # Write document to the document store to record the offset/length
            offset = doc_store_out.tell()
            text_bytes = document_text.encode('utf-8')
            length = len(text_bytes)
            doc_store_out.write(struct.pack('<Q', length))
            doc_store_out.write(text_bytes)

            # Write offsets incrementally to avoid storing them in memory
            doc_offsets_out.write(f"{doc_id}\t{offset}\t{length}\n")

            # Process the document text to get tokens
            tokens = process_document_text(document_text)

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

            # Update the postings cache and document frequency map with term_freq
            update_postings_cache(term_freq, doc_id)

            # spill if the cache exceeds max cache size
            if len(postings_cache) >= max_cache_size:
                # Update current file index 
                current_file_index += 1

                # write postings buffer and clear it
                write_postings_cache_to_file_on_disk(current_file_index, temp_file_prefix)

            doc_id += 1

    # spill remaining postings if exiting from while loop
    if postings_cache:
        current_file_index += 1
        write_postings_cache_to_file_on_disk(current_file_index, temp_file_prefix)

    # write to extra files (helpers use module-level OUTPUTS_DIR)
    save_doc_freqs()
    save_stats()
    save_page_table()

    # Close document store handlers
    doc_store_out.close()
    doc_offsets_out.close()

    print("1. Completed parsing stage.")

def argument_parser():
    '''
    Create the command line argument parser.
    '''
    parser = argparse.ArgumentParser(description="Parse a TSV file into temporary files")
    parser.add_argument("input", help="Path to TSV with lines: <passageID>\\t<passageText>")
    parser.add_argument("--prefix", default="data/index_postings_", help="Temporary postings file prefix")
    parser.add_argument("--buffer", type=int, default=DEFAULT_MAX_CACHE_SIZE, help="Max postings in RAM before spilling a sorted run")
    parser.add_argument("--outdir", default="out", help="Directory for stats files")
    return parser

if __name__ == "__main__":
    # parse command line arguments
    args = argument_parser().parse_args()

    # Parse the data
    parse_data(args.input, args.prefix, args.buffer, args.outdir)