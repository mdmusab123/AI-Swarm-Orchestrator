import json
import re
import sqlite3
import os
import uuid
from flask import Flask, request, Response, stream_with_context, render_template
import requests
from pyngrok import ngrok
from bs4 import BeautifulSoup
import pypdf
import chromadb
from werkzeug.utils import secure_filename

app = Flask(__name__)

# CONFIGURATION
MODEL = "gemma4:e2b"
API_URL = "http://127.0.0.1:11434/api/chat"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CHROMA DB SETUP (LOCAL RAG) ---
try:
    chroma_client = chromadb.PersistentClient(path="memory_db_vectors")
    docs_collection = chroma_client.get_or_create_collection(name="local_docs")
except Exception as e:
    print(f"ChromaDB Init Error: {e}. Document search will be unavailable.")

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        text = ""
        if filename.endswith('.pdf'):
            try:
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as e:
                return {"error": f"Failed to parse PDF: {e}"}, 500
        elif filename.endswith('.txt') or filename.endswith('.md'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                return {"error": f"Failed to read text file: {e}"}, 500
        else:
            return {"error": "Unsupported format. Use pdf, txt, or md."}, 400
            
        # Add to ChromaDB
        if text.strip():
            chunks = chunk_text(text)
            ids = [f"{filename}_{uuid.uuid4().hex[:8]}" for _ in chunks]
            metadatas = [{"source": filename} for _ in chunks]
            
            docs_collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            return {"message": f"Successfully processed and embedded {len(chunks)} chunks from {filename}."}
        else:
            return {"error": "File was empty or unreadable."}, 400

def search_docs(query):
    try:
        if docs_collection.count() == 0:
            return "No local documents have been uploaded yet."
            
        results = docs_collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant local documents found (or search query yielded no results)."
            
        res_strings = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            source = meta.get('source', 'Unknown')
            res_strings.append(f"--- Document: {source} (Snippet {i+1}) ---\n{doc}")
            
        return "\n\n".join(res_strings)
    except Exception as e:
        return f"Document search failed: {str(e)}"

# --- MEMORY DATABASE SETUP ---
DB_PATH = "memory.db"
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE
            )
        ''')
        conn.commit()

init_db()

def save_memory(fact):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO memory (fact) VALUES (?)", (fact.strip(),))
            conn.commit()
            return True
    except Exception as e:
        print(f"DB Error: {e}")
        return False

def get_memories():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact FROM memory")
            rows = cursor.fetchall()
            if rows:
                return [row[0] for row in rows[-20:]]
            return []
    except:
        return []

# --- WEB SEARCH ---
from bs4 import BeautifulSoup
import urllib.parse
def search_web(query):
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
        }
        data = {"q": query}
        r = requests.post(url, data=data, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        results = []
        for a in soup.find_all('a', class_='result__snippet'):
            results.append(a.text.strip())
            if len(results) >= 3:
                break
                
        res_strings = [f"- Snippet {i+1}: {text}" for i, text in enumerate(results)]
        return "\n".join(res_strings) if res_strings else "No web results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

# --- AGENTIC LOOP AND PARSER ---
def ask_ai_stream(messages, tools_enabled=True):
    if tools_enabled:
        memories = get_memories()
        if memories:
            memo_block = "User's saved long-term memories:\n" + "\n".join([f"- {m}" for m in memories])
            messages.insert(0, {"role": "system", "content": memo_block})

    def ollama_stream(msgs):
        try:
            response = requests.post(
                API_URL,
                json={"model": MODEL, "messages": msgs, "stream": True},
                stream=True,       
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    try:
                        data = json.loads(decoded)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        elif "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"[Ollama Connection Error: {str(e)}]"

    def generate():
        current_run_msgs = list(messages)
        loop_count = 0
        max_loops = 3

        while loop_count < max_loops:
            buffer = ""
            full_response = ""
            tool_triggered = False
            
            for chunk in ollama_stream(current_run_msgs):
                full_response += chunk
                buffer += chunk

                if '[' in buffer:
                    search_match = re.search(r'\[SEARCH:\s*(.*?)\]', buffer)
                    mem_save_match = re.search(r'\[MEM_SAVE:\s*(.*?)\]', buffer)
                    doc_search_match = re.search(r'\[SEARCH_DOC:\s*(.*?)\]', buffer)
                    
                    if search_match:
                        query = search_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🔍 Searching web for: {query}"}) + "\n"
                        tool_result = search_web(query)
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Web Search Results:\n{tool_result}\n\nUse this information to answer the initial query."})
                        tool_triggered = True
                        break
                        
                    elif mem_save_match:
                        fact = mem_save_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🧠 Saving memory..."}) + "\n"
                        save_memory(fact)
                        yield json.dumps({"type": "status", "text": f"✅ Memory saved."}) + "\n"
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Memory '{fact}' saved successfully. Acknowledge this."})
                        tool_triggered = True
                        break

                    elif doc_search_match:
                        query = doc_search_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"📚 Searching local documents for: {query}"}) + "\n"
                        tool_result = search_docs(query)
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Local Document Results:\n{tool_result}\n\nSynthesize this information to answer the initial query."})
                        tool_triggered = True
                        break
                    
                    if len(buffer) > 60:
                        idx = buffer.rfind('[')
                        if idx > 0:
                            yield json.dumps({"type": "content", "text": buffer[:idx]}) + "\n"
                            buffer = buffer[idx:]
                        else:
                            yield json.dumps({"type": "content", "text": buffer}) + "\n"
                            buffer = ""
                else:
                    yield json.dumps({"type": "content", "text": buffer}) + "\n"
                    buffer = ""

            if tool_triggered and loop_count < max_loops - 1:
                loop_count += 1
                continue
            else:
                if buffer:
                    clean_buf = re.sub(r'\[(?:SEARCH|MEM_SAVE|SEARCH_DOC).*$', '', buffer)
                    if clean_buf:
                        yield json.dumps({"type": "content", "text": clean_buf}) + "\n"
                break

    return generate()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    tools_enabled = data.get("tools_enabled", True)
    
    return Response(stream_with_context(ask_ai_stream(messages, tools_enabled)), mimetype='application/x-ndjson')

if __name__ == "__main__":
    try:
        public_url = ngrok.connect(5001).public_url
        print("\n" + "="*80)
        print(f"🚀 Nexus Engine is LIVE on the internet at: {public_url}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Ngrok connection warning: {e}")

    app.run(port=5001, debug=True, use_reloader=False, threaded=True)
