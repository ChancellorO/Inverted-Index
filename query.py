# query.py

import struct
import heapq
import argparse
import os

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
        
        # Read compressed data
        index_file.seek(start_offset)
        self.raw_data = index_file.read(length)
        
        # Parse header
        offset = 0
        term_size = struct.unpack('<Q', self.raw_data[offset:offset+8])[0]
        offset += 8
        offset += term_size  # skip term bytes
        
        self.num_blocks = struct.unpack('<Q', self.raw_data[offset:offset+8])[0]
        offset += 8
        
        # Parse block metadata
        self.blocks = []
        for _ in range(self.num_blocks):
            docIDs_size = struct.unpack('<Q', self.raw_data[offset:offset+8])[0]
            offset += 8
            freqs_size = struct.unpack('<Q', self.raw_data[offset:offset+8])[0]
            offset += 8
            
            docIDs_data = self.raw_data[offset:offset+docIDs_size]
            offset += docIDs_size
            freqs_data = self.raw_data[offset:offset+freqs_size]
            offset += freqs_size
            
            self.blocks.append({
                'docIDs_data': docIDs_data,
                'freqs_data': freqs_data,
                'decompressed_docIDs': None,
                'decompressed_freqs': None
            })
        
        self.current_block = -1
        self.current_index = -1
        self.current_docID = -1
        self.current_freq = 0
    
    def _decompress_block(self, block_idx):
        """Decompress a specific block."""
        if self.blocks[block_idx]['decompressed_docIDs'] is not None:
            return
        
        block = self.blocks[block_idx]
        
        # Decode docIDs (delta-encoded)
        docIDs = []
        offset = 0
        while offset < len(block['docIDs_data']):
            val, offset = varbyte_decode_one(block['docIDs_data'], offset)
            docIDs.append(val)
        
        # Undo delta encoding
        if docIDs:
            for i in range(1, len(docIDs)):
                docIDs[i] += docIDs[i-1]
        
        # Decode freqs
        freqs = []
        offset = 0
        while offset < len(block['freqs_data']):
            val, offset = varbyte_decode_one(block['freqs_data'], offset)
            freqs.append(val)
        
        block['decompressed_docIDs'] = docIDs
        block['decompressed_freqs'] = freqs
    
    def nextGEQ(self, k):
        """Find next docID >= k. Return docID or float('inf') if none."""
        while self.current_block < self.num_blocks:
            if self.current_block == -1:
                self.current_block = 0
                self.current_index = -1
            
            self._decompress_block(self.current_block)
            block = self.blocks[self.current_block]
            
            # Search within current block
            self.current_index += 1
            while self.current_index < len(block['decompressed_docIDs']):
                docID = block['decompressed_docIDs'][self.current_index]
                if docID >= k:
                    self.current_docID = docID
                    self.current_freq = block['decompressed_freqs'][self.current_index]
                    return docID
                self.current_index += 1
            
            # Move to next block
            self.current_block += 1
            self.current_index = -1
        
        return float('inf')
    
    def getFreq(self):
        """Get frequency of current posting."""
        return self.current_freq

class DocumentStore:
    def __init__(self, data_file, offset_file):
        self.data_file = data_file
        self.offsets = {}
        self.file_handle = None
        
        # Load offset index
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
    
    def get_text(self, docID):
        if docID not in self.offsets:
            return ""
        
        offset, length = self.offsets[docID]
        
        self.open()
        self.file_handle.seek(offset)
        
        length_bytes = self.file_handle.read(8)
        text_length = struct.unpack('<Q', length_bytes)[0]
        
        text_bytes = self.file_handle.read(text_length)
        return text_bytes.decode('utf-8')
    
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
    def __init__(self, index_file, lexicon_file, doc_freq_file, doc_len_file, stats_file, page_table_file, doc_store_file=None, doc_offsets_file=None):
        self.index_file_path = index_file
        self.lexicon = {}
        self.doc_freqs = {}
        self.doc_lengths = {}
        self.page_table = {}
        
        # Load lexicon
        with open(lexicon_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                term = parts[0]
                offset = int(parts[1])
                length = int(parts[2])
                doc_freq = int(parts[3])
                self.lexicon[term] = (offset, length, doc_freq)
        
        # Load doc frequencies
        with open(doc_freq_file, 'r') as f:
            for line in f:
                term, freq = line.strip().split()
                self.doc_freqs[term] = int(freq)
        
        # Load doc lengths
        with open(doc_len_file, 'r') as f:
            for line in f:
                docID, length = line.strip().split()
                self.doc_lengths[int(docID)] = int(length)
        
        # Load stats
        with open(stats_file, 'r') as f:
            line = f.readline().strip()
            self.total_docs, self.avg_doc_len = line.split('\t')
            self.total_docs = int(self.total_docs)
            self.avg_doc_len = float(self.avg_doc_len)
        
        # Load page table
        with open(page_table_file, 'r') as f:
            for line in f:
                docID, passage_id = line.strip().split(maxsplit=1)
                self.page_table[int(docID)] = passage_id
        
        # Add document store
        self.doc_store = None
        if doc_store_file and doc_offsets_file:
            if os.path.exists(doc_store_file) and os.path.exists(doc_offsets_file):
                self.doc_store = DocumentStore(doc_store_file, doc_offsets_file)
    
    def openList(self, term):
        """Open inverted list for term."""
        if term not in self.lexicon:
            return None
        offset, length, doc_freq = self.lexicon[term]
        index_file = open(self.index_file_path, 'rb')
        return InvertedList(term, offset, length, doc_freq, index_file)
    
    def closeList(self, inv_list):
        """Close inverted list."""
        if inv_list:
            inv_list.index_file.close()
    
    def bm25_score(self, term_freq, doc_len, doc_freq_term, k1=1.2, b=0.75):
        """Calculate BM25 score for a term in a document."""
        N = self.total_docs
        ft = doc_freq_term
        K = k1 * ((1 - b) + b * (doc_len / self.avg_doc_len))
        
        idf = ((N - ft + 0.5) / (ft + 0.5))
        if idf <= 0:
            idf = 0.01
        
        score = ((term_freq * (k1 + 1)) / (term_freq + K)) * (idf if idf > 0 else 0)
        return score

    def generate_snippet(self, text, query_terms, window_size=50, max_snippets=2):
        """Generate query-dependent snippet from text."""
        words = text.split()
        
        positions = []
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?;:')
            if word_clean in query_terms:
                positions.append(i)
        
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
                if w.lower().strip('.,!?;:') in query_terms:
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
    
    def disjunctive_query(self, query_terms, k=10, with_snippets=False):
        """Disjunctive (OR) query using DAAT."""
        lists = []
        for term in query_terms:
            inv_list = self.openList(term)
            if inv_list:
                lists.append(inv_list)
        
        if not lists:
            return []
        
        doc_scores = {}
        
        # Use heap to merge lists
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
            
            # Advance this list
            next_docID = inv_list.nextGEQ(docID + 1)
            if next_docID != float('inf'):
                heapq.heappush(heap, (next_docID, list_idx))
        
        for inv_list in lists:
            self.closeList(inv_list)
        
        # Get top k
        top_k = heapq.nlargest(k, doc_scores.items(), key=lambda x: x[1])
   
        if with_snippets and self.doc_store:
            docIDs = [docID for docID, score in top_k]
            texts = self.doc_store.get_texts_batch(docIDs)
            
            results = []
            for docID, score in top_k:
                text = texts.get(docID, "")
                snippet = self.generate_snippet(text, query_terms)
                results.append((self.page_table[docID], score, snippet))
            return results
        else:
            return [(self.page_table[docID], score) for docID, score in top_k]
    
    def conjunctive_query(self, query_terms, k=10, with_snippets=False):
        """Conjunctive (AND) query using DAAT."""
        lists = []
        for term in query_terms:
            inv_list = self.openList(term)
            if inv_list:
                lists.append(inv_list)
            else:
                return []  # term not in index
        
        if not lists:
            return []
        
        doc_scores = {}
        
        # Initialize all lists
        docIDs = [inv_list.nextGEQ(0) for inv_list in lists]
        
        while all(d != float('inf') for d in docIDs):
            max_docID = max(docIDs)
            
            # Try to align all lists to max_docID
            aligned = True
            for i, inv_list in enumerate(lists):
                if docIDs[i] < max_docID:
                    docIDs[i] = inv_list.nextGEQ(max_docID)
                    if docIDs[i] != max_docID:
                        aligned = False
                        break
            
            if aligned and all(d == max_docID for d in docIDs):
                # All lists have this docID
                score = 0.0
                doc_len = self.doc_lengths.get(max_docID, self.avg_doc_len)
                
                for inv_list in lists:
                    freq = inv_list.getFreq()
                    score += self.bm25_score(freq, doc_len, inv_list.doc_freq)
                
                doc_scores[max_docID] = score
                
                # Advance all lists
                docIDs = [inv_list.nextGEQ(max_docID + 1) for inv_list in lists]
        
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
                results.append((self.page_table[docID], score, snippet))
            return results
        else:
            return [(self.page_table[docID], score) for docID, score in top_k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='tmp/index.bin')
    parser.add_argument('--lexicon', default='tmp/lexicon.txt')
    parser.add_argument('--doc-freq', default='tmp/doc_frequencies.txt')
    parser.add_argument('--doc-len', default='tmp/document_lengths.txt')
    parser.add_argument('--stats', default='tmp/collection_stats.txt')
    parser.add_argument('--page-table', default='tmp/page_table.txt')
    parser.add_argument('--doc-store', default='tmp/documents.dat')
    parser.add_argument('--doc-offsets', default='tmp/doc_offsets.txt')
    args = parser.parse_args()
    
    qp = QueryProcessor(args.index, args.lexicon, args.doc_freq, args.doc_len, args.stats, args.page_table, args.doc_store, args.doc_offsets)
    
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
        
        if mode == 'OR':
            results = qp.disjunctive_query(terms)
        elif mode == 'AND':
            results = qp.conjunctive_query(terms)
        else:
            print("Unknown mode. Use OR or AND")
            continue
        
        print(f"\nTop {len(results)} results:")
        for i, (passage_id, score) in enumerate(results, 1):
            print(f"{i}. {passage_id} (score: {score:.4f})")

if __name__ == '__main__':
    main()