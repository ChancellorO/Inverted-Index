import argparse
import glob
import heapq
import os
import struct
from typing import List

def varbyte_encode(number: int) -> bytes:
    """Variable-byte encode a non-negative integer and return bytes."""
    if number < 0:
        raise ValueError("varbyte_encode expects non-negative integers")
    encoded = bytearray()
    while True:
        byte = number & 0x7F
        number >>= 7
        if number == 0:
            # set continuation bit on last byte
            encoded.append(byte | 0x80)
            break
        else:
            encoded.append(byte)
    return bytes(encoded)


def merge_inverted_indexes(index_files: List[str], output_index_file: str, output_lexicon_file: str, block_size: int = 128):
    '''
    Merge multiple sorted posting files into a single compressed inverted index and lexicon.
    '''

    # Open all temporary posting files
    input_files = []
    for path in index_files:
        file = open(path, "r", encoding="utf-8")
        input_files.append(file)

    # Heap entries: (term, docID, freq, fileIndex)
    heap = []

    # Read first posting from each file
    for idx, file in enumerate(input_files):
        line = file.readline()

        # skip empty lines
        if not line:
            continue

        # parse line
        parts = line.rstrip("\n").split()

        # skip malformed lines
        if len(parts) < 3:
            continue
        
        term = parts[0]
        docID = int(parts[1])
        freq = int(parts[2])

        # push to heap
        heapq.heappush(heap, (term, docID, freq, idx))

    # Prepare output files
    os.makedirs(os.path.dirname(output_index_file) or ".", exist_ok=True)
    out_file = open(output_index_file, "wb")

    os.makedirs(os.path.dirname(output_lexicon_file) or ".", exist_ok=True)
    lexicon_out = open(output_lexicon_file, "w", encoding="utf-8")

    # Variables to track current term postings
    current_term = None
    docIDs = []
    freqs = []

    # Merge process
    while heap:
        # Get smallest term posting
        term, docID, freq, file_idx = heapq.heappop(heap)

        # If we've moved to a new term, flush previous
        if current_term is None:
            current_term = term

        if term != current_term:
            # write postings for current_term
            _write_term_postings(out_file, lexicon_out, current_term, docIDs, freqs, block_size)
            docIDs = []
            freqs = []
            current_term = term

        docIDs.append(docID)
        freqs.append(freq)

        # Read next line from same file and push to heap
        next_line = input_files[file_idx].readline()
        if next_line:
            parts = next_line.rstrip("\n").split()
            if len(parts) >= 3:
                nterm = parts[0]
                try:
                    ndocID = int(parts[1])
                    nfreq = int(parts[2])
                    heapq.heappush(heap, (nterm, ndocID, nfreq, file_idx))
                except ValueError:
                    pass

    # Flush last term
    if current_term is not None and docIDs:
        _write_term_postings(out_file, lexicon_out, current_term, docIDs, freqs, block_size)

    # Close files
    for f in input_files:
        f.close()
    out_file.close()
    lexicon_out.close()

    print("[INFO] Merged inverted index and lexicon generated successfully.")


def _write_term_postings(out_file, lexicon_out, term: str, docIDs: List[int], freqs: List[int], block_size: int):
    """Write postings for a single term into out_file (binary) and append lexicon entry.

    Format (binary):
      - termSize (8 bytes, unsigned little-endian)
      - term bytes
      - numBlocks (8 bytes unsigned)
      For each block:
        - docIDsSize (8 bytes unsigned)
        - freqsSize (8 bytes unsigned)
        - encodedDocIDs (varbyte bytes)
        - encodedFreqs (varbyte bytes)

    The lexicon line is: term startOffset length docFrequency (text)
    """
    BLOCK = block_size
    # record start offset
    term_start_offset = out_file.tell()

    term_bytes = term.encode("utf-8")
    term_size = len(term_bytes)
    # write term size as unsigned 8-byte
    out_file.write(struct.pack('<Q', term_size))
    out_file.write(term_bytes)

    num_blocks = (len(docIDs) + BLOCK - 1) // BLOCK
    out_file.write(struct.pack('<Q', num_blocks))

    for block_index in range(num_blocks):
        start = block_index * BLOCK
        end = min(start + BLOCK, len(docIDs))
        block_docIDs = docIDs[start:end]
        block_freqs = freqs[start:end]

        # delta encode docIDs within block
        delta_docIDs = []
        if block_docIDs:
            prev = block_docIDs[0]
            delta_docIDs.append(prev)
            for d in block_docIDs[1:]:
                delta_docIDs.append(d - prev)
                prev = d

        # varbyte encode
        encoded_docIDs = bytearray()
        for v in delta_docIDs:
            encoded_docIDs.extend(varbyte_encode(v))

        encoded_freqs = bytearray()
        for f in block_freqs:
            encoded_freqs.extend(varbyte_encode(f))

        docIDs_size = len(encoded_docIDs)
        freqs_size = len(encoded_freqs)

        out_file.write(struct.pack('<Q', docIDs_size))
        out_file.write(struct.pack('<Q', freqs_size))

        if docIDs_size:
            out_file.write(bytes(encoded_docIDs))
        if freqs_size:
            out_file.write(bytes(encoded_freqs))

    new_offset = out_file.tell()
    length = new_offset - term_start_offset
    doc_frequency = len(docIDs)

    # lexicon line: term offset length docFrequency
    lexicon_out.write(f"{term} {term_start_offset} {length} {doc_frequency}\n")


def argument_parser():
    parser = argparse.ArgumentParser(description="Merge sorted posting runs into a compressed inverted index and lexicon.")
    parser.add_argument('--inputs', required=True, help='Glob pattern for input run files, e.g. "tmp/run_*.txt"')
    parser.add_argument('--output-index', default='output/index.bin', help='Output binary index file')
    parser.add_argument('--output-lexicon', default='output/lexicon.txt', help='Output lexicon (text) file')
    parser.add_argument('--block-size', type=int, default=128, help='Number of postings per block')
    return parser


if __name__ == '__main__':
    # Parse arguments
    args = argument_parser().parse_args()

    # Find input files
    files = sorted(glob.glob(args.inputs))

    # Check if any files found
    if not files:
        print(f"No input files matched pattern provided: {args.inputs}")
        raise SystemExit(1)
    
    merge_inverted_indexes(files, args.output_index, args.output_lexicon, block_size=args.block_size)
