from flask import Flask, render_template, request, jsonify
from query import QueryProcessor
import os

app = Flask(__name__)

# Initialize query processor with document store
qp = QueryProcessor(
    index_file='tmp/index.bin',
    lexicon_file='tmp/lexicon.txt',
    doc_freq_file='tmp/doc_frequencies.txt',
    doc_len_file='tmp/document_lengths.txt',
    stats_file='tmp/collection_stats.txt',
    page_table_file='tmp/page_table.txt',
    doc_store_file='tmp/documents.dat',
    doc_offsets_file='tmp/doc_offsets.txt'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query', '').strip().lower()
    mode = data.get('mode', 'OR').upper()
    
    if not query_text:
        return jsonify({'error': 'Empty query'}), 400
    
    terms = query_text.split()
    
    if mode == 'OR':
        results = qp.disjunctive_query(terms, k=10, with_snippets=True)
    elif mode == 'AND':
        results = qp.conjunctive_query(terms, k=10, with_snippets=True)
    else:
        return jsonify({'error': 'Invalid mode'}), 400
    
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
        'total': len(formatted_results)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)