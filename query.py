# query.py
import struct
import heapq
import argparse
import os
import math 
import time

def varbyte_decode_one(data, offset):
    """Decode one varbyte integer from data starting at offset."""
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte & 0x80:
            return result, offset
        shift += 7
    return result, offset


class InvertedList:
    """Represents an opened inverted list for a term."""
    def __init__(self, term, start_offset, length, doc_freq, index_file):
        self.term = term
        self.start_offset = start_offset
        self.length = length
        self.doc_freq = doc_freq
        self.index_file = index_file
        
        index_file.seek(start_offset)
        term_size = struct.unpack('<Q', index_file.read(8))[0]
        index_file.seek(term_size, 1)  # Skip term bytes
        self.num_blocks = struct.unpack('<Q', index_file.read(8))[0]
        
        # Store block locations without reading data
        self.block_metadata = []
        for _ in range(self.num_blocks):
            block_start = index_file.tell()
            max_docID = struct.unpack('<Q', index_file.read(8))[0]
            docIDs_size = struct.unpack('<Q', index_file.read(8))[0]
            freqs_size = struct.unpack('<Q', index_file.read(8))[0]
            self.block_metadata.append({
                'offset': block_start,
                'max_docID': max_docID,
                'docIDs_size': docIDs_size,
                'freqs_size': freqs_size,
                'decompressed_docIDs': None,
                'decompressed_freqs': None
            })
            # Skip the actual data
            index_file.seek(docIDs_size + freqs_size, 1)
        
        self.current_block = -1
        self.current_index = -1
        self.current_docID = -1
        self.current_freq = 0
    
    def _decompress_block(self, block_idx):
        block = self.block_metadata[block_idx]
        
        # Check if already decompressed
        if block['decompressed_docIDs'] is not None:
            return
        
        # NOW read from disk only when needed
        self.index_file.seek(block['offset'] + 24)  # Skip maxDocID (8 bytes) + two size fields (16 bytes)
        docIDs_data = self.index_file.read(block['docIDs_size'])
        freqs_data = self.index_file.read(block['freqs_size'])
        
        # Decompress docIDs (with delta decoding)
        docIDs = []
        offset = 0
        while offset < len(docIDs_data):
            val, offset = varbyte_decode_one(docIDs_data, offset)
            docIDs.append(val)
        
        if docIDs:
            for i in range(1, len(docIDs)):
                docIDs[i] += docIDs[i-1]
        
        # Decompress freqs
        freqs = []
        offset = 0
        while offset < len(freqs_data):
            val, offset = varbyte_decode_one(freqs_data, offset)
            freqs.append(val)
        
        block['decompressed_docIDs'] = docIDs
        block['decompressed_freqs'] = freqs
    
    def nextGEQ(self, k):
        while self.current_block < self.num_blocks:
            if self.current_block == -1:
                self.current_block = 0
                self.current_index = -1
                # self.current_offset = 0  # Track position in compressed data
            
            block = self.block_metadata[self.current_block]

            if k > block['max_docID']:
                self.current_block += 1
                continue
            
            # Decompress block only when needed
            self._decompress_block(self.current_block)
            
            docIDs = block['decompressed_docIDs']
            freqs = block['decompressed_freqs']
            
            # Binary search within block
            left, right = 0, len(docIDs) - 1
            result_idx = -1
            
            while left <= right:
                mid = (left + right) // 2
                if docIDs[mid] >= k:
                    result_idx = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            if result_idx != -1:
                self.current_index = result_idx
                self.current_docID = docIDs[result_idx]
                self.current_freq = freqs[result_idx]
                return self.current_docID
            
            # Move to next block
            self.current_block += 1
        
        return float('inf')
        
    def getFreq(self):
        return self.current_freq


class DocumentStore:
    def __init__(self, data_file, offset_file):
        self.data_file = data_file
        self.offsets = {}
        self.file_handle = None
        
        with open(offset_file, 'r') as f:
            for line in f:
                docID, offset, length = line.strip().split('\t')
                self.offsets[int(docID)] = (int(offset), int(length))
    
    def open(self):
        if self.file_handle is None:
            self.file_handle = open(self.data_file, 'rb')
    
    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def get_texts_batch(self, docIDs):
        results = {}
        self.open()
        
        sorted_docs = sorted([(self.offsets[d], d) for d in docIDs if d in self.offsets])
        
        for (offset, length), docID in sorted_docs:
            self.file_handle.seek(offset)
            length_bytes = self.file_handle.read(8)
            text_length = struct.unpack('<Q', length_bytes)[0]
            text_bytes = self.file_handle.read(text_length)
            results[docID] = text_bytes.decode('utf-8')
        
        return results


class QueryProcessor:
    def __init__(self, index_file, lexicon_file, doc_freq_file, stats_file, page_table_file, doc_store_file=None, doc_offsets_file=None):
        self.index_file_path = index_file
        self.lexicon = {}
        self.doc_freqs = {}
        self.doc_lengths = {}
        
        with open(lexicon_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                term = parts[0]
                offset = int(parts[1])
                length = int(parts[2])
                doc_freq = int(parts[3])
                self.lexicon[term] = (offset, length, doc_freq)
        
        with open(doc_freq_file, 'r') as f:
            for line in f:
                term, freq = line.strip().split()
                self.doc_freqs[term] = int(freq)
        
        with open(stats_file, 'r') as f:
            line = f.readline().strip()
            self.total_docs, self.avg_doc_len = line.split('\t')
            self.total_docs = int(self.total_docs)
            self.avg_doc_len = float(self.avg_doc_len)
        
        with open(page_table_file, 'r') as f:
            for line in f:
                docID, length = line.strip().split()
                self.doc_lengths[int(docID)] = int(length)
        
        self.doc_store = None
        if doc_store_file and doc_offsets_file:
            if os.path.exists(doc_store_file) and os.path.exists(doc_offsets_file):
                self.doc_store = DocumentStore(doc_store_file, doc_offsets_file)
    
    def openList(self, term):
        if term not in self.lexicon:
            return None
        offset, length, doc_freq = self.lexicon[term]
        index_file = open(self.index_file_path, 'rb')
        return InvertedList(term, offset, length, doc_freq, index_file)
    
    def closeList(self, inv_list):
        if inv_list:
            inv_list.index_file.close()
    
    def bm25_score(self, term_freq, doc_len, doc_freq_term, k1=1.2, b=0.75):
        N = self.total_docs
        ft = doc_freq_term
        K = k1 * ((1 - b) + b * (doc_len / self.avg_doc_len))
        
        idf = math.log((N - ft + 0.5) / (ft + 0.5))
        if idf <= 0:
            idf = 0.01
        
        score = ((term_freq * (k1 + 1)) / (term_freq + K)) * idf
        return score

    def generate_snippet(self, text, query_terms, window_size=50, max_snippets=2):
        words = text.split()
        
        positions = []
        for i, word in enumerate(words):
            word_lower = word.lower()
            for term in query_terms:
                if term in word_lower:
                    positions.append(i)
                    break
        
        if not positions:
            snippet = ' '.join(words[:window_size])
            return snippet + '...' if len(words) > window_size else snippet
        
        snippets = []
        used_positions = set()
        
        for pos in sorted(positions):
            if pos in used_positions:
                continue
            
            start = max(0, pos - window_size // 2)
            end = min(len(words), pos + window_size // 2)
            
            snippet_words = words[start:end]
            
            highlighted = []
            for w in snippet_words:
                w_lower = w.lower()
                if any(term in w_lower for term in query_terms):
                    highlighted.append(f"**{w}**")
                else:
                    highlighted.append(w)
            
            snippet = ' '.join(highlighted)
            if start > 0:
                snippet = '...' + snippet
            if end < len(words):
                snippet = snippet + '...'
            
            snippets.append(snippet)
            used_positions.update(range(start, end))
            
            if len(snippets) >= max_snippets:
                break
        
        return ' '.join(snippets)
    
    def disjunctive_query(self, query_terms, k=100, with_snippets=False):
        lists = []
        for term in query_terms:
            inv_list = self.openList(term)
            if inv_list:
                lists.append(inv_list)
        
        if not lists:
            return []
        
        doc_scores = {}
        heap = []
        
        for i, inv_list in enumerate(lists):
            docID = inv_list.nextGEQ(0)
            if docID != float('inf'):
                heapq.heappush(heap, (docID, i))
        
        while heap:
            docID, list_idx = heapq.heappop(heap)
            
            if docID not in doc_scores:
                doc_scores[docID] = 0.0
            
            inv_list = lists[list_idx]
            freq = inv_list.getFreq()
            doc_len = self.doc_lengths.get(docID, self.avg_doc_len)
            score = self.bm25_score(freq, doc_len, inv_list.doc_freq)
            doc_scores[docID] += score
            
            next_docID = inv_list.nextGEQ(docID + 1)
            if next_docID != float('inf'):
                heapq.heappush(heap, (next_docID, list_idx))
        
        for inv_list in lists:
            self.closeList(inv_list)
        
        top_k = heapq.nlargest(k, doc_scores.items(), key=lambda x: x[1])
   
        if with_snippets and self.doc_store:
            docIDs = [docID for docID, score in top_k]
            texts = self.doc_store.get_texts_batch(docIDs)
            
            results = []
            for docID, score in top_k:
                text = texts.get(docID, "")
                snippet = self.generate_snippet(text, query_terms)
                results.append((docID, score, snippet))
            return results
        else:
            return [(docID, score) for docID, score in top_k]
    
    def conjunctive_query(self, query_terms, k=100, with_snippets=False):
        lists = []
        for term in query_terms:
            inv_list = self.openList(term)
            if inv_list:
                lists.append(inv_list)
            else:
                return []
        
        if not lists:
            return []
        
        doc_scores = {}
        docIDs = [inv_list.nextGEQ(0) for inv_list in lists]
        
        while all(d != float('inf') for d in docIDs):
            max_docID = max(docIDs)
            
            if all(d == max_docID for d in docIDs):
                score = 0.0
                doc_len = self.doc_lengths.get(max_docID, self.avg_doc_len)
                
                for inv_list in lists:
                    freq = inv_list.getFreq()
                    score += self.bm25_score(freq, doc_len, inv_list.doc_freq)
                
                doc_scores[max_docID] = score
                docIDs = [inv_list.nextGEQ(max_docID + 1) for inv_list in lists]
            else:
                for i in range(len(lists)):
                    if docIDs[i] < max_docID:
                        docIDs[i] = lists[i].nextGEQ(max_docID)
        
        for inv_list in lists:
            self.closeList(inv_list)
        
        top_k = heapq.nlargest(k, doc_scores.items(), key=lambda x: x[1])
        
        if with_snippets and self.doc_store:
            docIDs = [docID for docID, score in top_k]
            texts = self.doc_store.get_texts_batch(docIDs)
            
            results = []
            for docID, score in top_k:
                text = texts.get(docID, "")
                snippet = self.generate_snippet(text, query_terms)
                results.append((docID, score, snippet))
            return results
        else:
            return [(docID, score) for docID, score in top_k]


def run_cli(qp):
    print("Query Processor Ready. Type 'quit' to exit.")
    print("Commands: OR <terms> | AND <terms>")

    while True:
        query = input("\nQuery> ").strip()
        if query.lower() == 'quit':
            break

        parts = query.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: OR term1 term2 ... | AND term1 term2 ...")
            continue

        mode = parts[0].upper()
        terms = parts[1].lower().split()

        # Ask for the number of results
        try:
            k = int(input("Enter the number of results to display: ").strip())
        except ValueError:
            print("Invalid number. Using default (100).")
            k = 100

        # Start timing
        start_time = time.time()

        if mode == 'OR':
            results = qp.disjunctive_query(terms, k=k)
        elif mode == 'AND':
            results = qp.conjunctive_query(terms, k=k)
        else:
            print("Unknown mode. Use OR or AND")
            continue

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Search Time ({elapsed_time:.4f} seconds):")

        print(f"\nTop {len(results)} results:")
        for i, (passage_id, score) in enumerate(results, 1):
            print(f"{i}. {passage_id} (score: {score:.4f})")


def run_web(qp):
    from flask import Flask, render_template, request, jsonify

    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/search', methods=['POST'])
    def search():
        data = request.json
        query_text = data.get('query', '').strip().lower()
        mode = data.get('mode', 'OR').upper()
        k = int(data.get('results', 100))  # Default to 100 results if not provided

        if not query_text:
            return jsonify({'error': 'Empty query'}), 400

        terms = query_text.split()
        # Start timing
        start_time = time.time()

        if mode == 'OR':
            results = qp.disjunctive_query(terms, k=k, with_snippets=True)
        elif mode == 'AND':
            results = qp.conjunctive_query(terms, k=k, with_snippets=True)
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        formatted_results = [
            {
                'rank': i + 1,
                'passage_id': passage_id,
                'score': round(score, 4),
                'snippet': snippet
            }
            for i, (passage_id, score, snippet) in enumerate(results)
        ]

        return jsonify({
            'query': query_text,
            'mode': mode,
            'results': formatted_results,
            'total': len(formatted_results),
            'time_seconds': round(elapsed_time, 4)
        })

    app.run(debug=True, port=5000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cli', 'web'], default='cli', help='Run mode')
    parser.add_argument('--index', default='out/index.bin')
    parser.add_argument('--lexicon', default='out/lexicon.txt')
    parser.add_argument('--doc-freq', default='out/document_frequencies.txt')
    parser.add_argument('--stats', default='out/stats.txt')
    parser.add_argument('--page-table', default='out/page_table.txt')
    parser.add_argument('--doc-store', default='out/documents.dat')
    parser.add_argument('--doc-offsets', default='out/doc_offsets.txt')
    parser.add_argument('--results', type=int, default=100, help='Number of results to display')
    args = parser.parse_args()

    qp = QueryProcessor(
        args.index, args.lexicon, args.doc_freq,
        args.stats, args.page_table,
        args.doc_store, args.doc_offsets
    )

    if args.mode == 'cli':
        run_cli(qp)
    else:
        run_web(qp)


if __name__ == '__main__':
    main()