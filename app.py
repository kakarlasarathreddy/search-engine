from flask import Flask, render_template, request
import chromadb
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection('subtitles')

model = SentenceTransformer('all-mpnet-base-v2')

def query_chromadb(query):
    query_embeddings = model.encode(query)
    result = collection.query(
        query_embeddings=query_embeddings.tolist(),
        n_results=10,
        include=['documents']
    )
    transformed_result = []
    for doc_num, chunk_list in enumerate(result['documents'], start=1):
        document_chunks = []
        for i, chunk in enumerate(chunk_list, start=1):
            transformed_chunk = transform_chunk(chunk)
            document_chunks.append(transformed_chunk)
        transformed_result.append((doc_num, document_chunks))
    return transformed_result

def transform_chunk(chunk):
    transformed_chunk = ' '.join(''.join(token.split('#')) for token in chunk.split())
    sentences = transformed_chunk.split('.')
    transformed_chunk = '. '.join(sentences) + '.'  
    return transformed_chunk

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    relevant_output = query_chromadb(query)
    return render_template('results.html', relevant_output=relevant_output)

if __name__ == '__main__':
    app.run(debug=True)