import json
import platform
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
    memory_collection = chroma_client.get_or_create_collection(name="long_term_memory")
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

import time

def save_memory(fact):
    try:
        memory_collection.add(
            documents=[fact.strip()],
            metadatas=[{"timestamp": time.time()}],
            ids=[f"mem_{uuid.uuid4().hex[:8]}"]
        )
        return True
    except Exception as e:
        print(f"Memory DB Error: {e}")
        return False

def retrieve_relevant_memories(query_text):
    try:
        if memory_collection.count() == 0:
            return []
        results = memory_collection.query(
            query_texts=[query_text],
            n_results=5
        )
        if results and results['documents'] and results['documents'][0]:
            return results['documents'][0]
        return []
    except:
        return []

# --- WEB SEARCH & SCRAPING ---
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
        for d in soup.find_all('div', class_='result'):
            a_url = d.find('a', class_='result__url')
            a_snippet = d.find('a', class_='result__snippet')
            if a_url and a_snippet:
                href = a_url.get('href', '').strip()
                # DuckDuckGo sometimes wraps hrefs, unquote them
                if href.startswith('//duckduckgo.com/l/?uddg='):
                    href = urllib.parse.unquote(href.split('uddg=')[1].split('&')[0])
                snippet = a_snippet.text.strip()
                results.append((href, snippet))
            if len(results) >= 3:
                break
                
        res_strings = [f"- Source [URL]: {url} \n  Snippet: {text}" for url, text in results]
        return "\n\n".join(res_strings) if res_strings else "No web results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

def read_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        # Squeeze out multiple spaces/newlines
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Limit to 5000 characters
        if len(text) > 5000:
            text = text[:5000] + "... [Content Truncated]"
            
        return text if text else "No readable text found on page."
    except Exception as e:
        return f"Failed to read URL: {str(e)}"

# --- LOCAL PYTHON EXECUTION ---
import sys, io, traceback
def execute_python(code_str):
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        exec(code_str, {})
        output = captured_output.getvalue()
        return output if output.strip() else "Executed successfully but there was no printed output."
    except Exception as e:
        tb = traceback.format_exc()
        return f"Python Execution Error:\n{tb}"
    finally:
        sys.stdout = old_stdout

# --- LOCAL SYSTEM SHELL EXECUTION ---
import subprocess
def execute_shell(command):
    try:
        # Detect OS for cross-platform support
        is_windows = platform.system() == "Windows"
        shell_cmd = ["powershell", "-Command", command] if is_windows else ["/bin/bash", "-c", command]
        
        result = subprocess.run(shell_cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output if output.strip() else "Command executed successfully with no output."
    except Exception as e:
        return f"Shell Execution Error: {str(e)}"

# --- AGENTIC LOOP AND PARSER ---
def ask_ai_stream(messages, target_model=MODEL, tools_enabled=True, router_enabled=True, force_web_search=False):
    if tools_enabled:
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        if user_msgs:
            last_query = user_msgs[-1]
            memories = retrieve_relevant_memories(last_query)
            if memories:
                memo_block = "User's relevant long-term memories regarding this context:\n" + "\n".join([f"- {m}" for m in memories])
                messages.insert(0, {"role": "system", "content": memo_block})

    def ollama_stream(msgs):
        try:
            response = requests.post(
                API_URL,
                json={"model": target_model, "messages": msgs, "stream": True},
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
        
        # --- ROUTER AGENT LOGIC ---
        if tools_enabled:
            if not router_enabled:
                os_type = "powershell" if platform.system() == "Windows" else "bash"
                tool_instructions = f"""You have tools enabled. You MUST use one of these tools if requested:
To search the web: [SEARCH: query]
To read the full text of an article: [READ_URL: https://...]
To save memory: [MEM_SAVE: fact]
To search within uploaded user documents: [SEARCH_DOC: query]
To run python code: [PYTHON: print("hello")]
To run {os_type} system commands (WAIT for user permission!): [RUN_SHELL: ping google.com]
Only output one tool per response."""
            else:
                user_msgs = [m["content"] for m in current_run_msgs if m["role"] == "user"]
                last_msg = user_msgs[-1] if user_msgs else "General query"
                    
                router_prompt = f"""You are a routing agent. Read the user's message.
Categorize the request into EXACTLY ONE of these strings based on what it needs:
[ROUTE: RESEARCH] - Requires searching the internet or reading URLs to get current info.
[ROUTE: CODE] - Requires mathematical calculation, logic, data analysis, or executing python code.
[ROUTE: DOCS] - Requires searching uploaded documents.
[ROUTE: SYSTEM] - Requires executing local system shell commands (like powershell), managing files, or getting OS info.
[ROUTE: GENERAL] - General conversation or simple memory saving.

User Message: {last_msg}

Output ONLY the exact category string and nothing else. Example: [ROUTE: GENERAL]"""

                if force_web_search:
                    route_text = "[ROUTE: RESEARCH]"
                else:
                    try:
                        router_res = requests.post(API_URL, json={
                            "model": target_model, 
                            "messages": [{"role": "user", "content": router_prompt}], 
                            "stream": False
                        }, timeout=10)
                        router_data = router_res.json()
                        route_text = router_data.get("message", {}).get("content", "").strip()
                    except Exception as e:
                        route_text = "[ROUTE: GENERAL]" # fallback
                    
                if "[ROUTE: RESEARCH]" in route_text:
                    cat = "RESEARCH"
                    icon = "🔍 Explorer Node"
                    tool_instructions = "You are the Explorer Node. You MUST use these tools if necessary:\nTo search the web: [SEARCH: query]\nTo read an article: [READ_URL: url]\nOutput only one tool per response."
                elif "[ROUTE: CODE]" in route_text:
                    cat = "CODE"
                    icon = "🤖 Coder Node"
                    tool_instructions = "You are the Coder Node. You MUST use this tool to calculate or analyze:\nTo run python code: [PYTHON: print('hello')]\nOutput only one tool per response."
                elif "[ROUTE: DOCS]" in route_text:
                    cat = "DOCS"
                    icon = "📚 Document Search Node"
                    tool_instructions = "You are the Document Node. You MUST use this tool to answer file questions:\nTo search documents: [SEARCH_DOC: query]\nOutput only one tool per response."
                elif "[ROUTE: SYSTEM]" in route_text:
                    cat = "SYSTEM"
                    icon = "💻 Administrator Node"
                    os_type = "powershell" if platform.system() == "Windows" else "bash"
                    tool_instructions = f"You are the System Administrator Node. You MUST use this tool to manage the computer natively:\nTo run {os_type} commands: [RUN_SHELL: ping google.com]\nWait for the user's explicit permission. Output only one tool per response."
                else:
                    cat = "GENERAL"
                    icon = "🧠 General Node"
                    tool_instructions = "You are the General Node. You MUST use this tool if explicitly asked to remember something:\nTo save memory: [MEM_SAVE: fact]\nOutput only one tool per response."
                
            # Inject dynamic sub-agent instructions safely into system prompt
            if len(current_run_msgs) > 0 and current_run_msgs[0]["role"] == "system":
                current_run_msgs[0]["content"] += "\n\n" + tool_instructions
            else:
                current_run_msgs.insert(0, {"role": "system", "content": tool_instructions})
        
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
                    read_url_match = re.search(r'\[READ_URL:\s*(.*?)\]', buffer)
                    python_match = re.search(r'\[PYTHON:\s*(.*?)\n?\]', buffer, re.DOTALL)
                    shell_match = re.search(r'\[RUN_SHELL:\s*(.*?)\n?\]', buffer, re.DOTALL)
                    
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

                    elif read_url_match:
                        url = read_url_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🌐 Reading webpage: {url}"}) + "\n"
                        tool_result = read_url(url)
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Extracted Website Application Text:\n{tool_result}\n\nSynthesize this information to answer the user's initial query."})
                        tool_triggered = True
                        break

                    elif python_match:
                        code = python_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"⚙️ Executing Python Script..."}) + "\n"
                        # Extra visual flourish: yield the code to the user chat UI too so they see what runs
                        yield json.dumps({"type": "content", "text": f"\n```python\n{code}\n```\n"}) + "\n"
                        
                        tool_result = execute_python(code)
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Python Execution Output (stdout/stderr):\n{tool_result}\n\nUse this information to answer the initial query."})
                        tool_triggered = True
                        break

                    elif shell_match:
                        code = shell_match.group(1).strip()
                        yield json.dumps({"type": "action_request", "action": "RUN_SHELL", "command": code}) + "\n"
                        # Terminate the stream explicitly. The frontend will pick up execution.
                        break
                    
                    if len(buffer) > 60:
                        if re.match(r'^\[(?:SEARCH|MEM_SAVE|SEARCH_DOC|READ_URL|PYTHON|RUN_SHELL):', buffer):
                            if len(buffer) > 5000: # safety bailout
                                yield json.dumps({"type": "content", "text": buffer}) + "\n"
                                buffer = ""
                        else:
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
                    clean_buf = re.sub(r'\[(?:SEARCH|MEM_SAVE|SEARCH_DOC|READ_URL|PYTHON|RUN_SHELL).*$', '', buffer, flags=re.DOTALL)
                    if clean_buf:
                        yield json.dumps({"type": "content", "text": clean_buf}) + "\n"
                break

    return generate()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/models", methods=["GET"])
def get_models():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/execute_shell", methods=["POST"])
def run_shell():
    data = request.json or {}
    command = data.get("command", "")
    if not command:
        return {"error": "No command provided"}, 400
    
    output = execute_shell(command)
    return {"output": output}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    tools_enabled = data.get("tools_enabled", True)
    target_model = data.get("model", MODEL)
    router_enabled = data.get("router_enabled", True)
    force_web_search = data.get("force_web_search", False)
    
    return Response(stream_with_context(ask_ai_stream(messages, target_model, tools_enabled, router_enabled, force_web_search)), mimetype='application/x-ndjson')

if __name__ == "__main__":
    try:
        public_url = ngrok.connect(5001).public_url
        print("\n" + "="*80)
        print(f"🚀 Nexus Engine is LIVE on the internet at: {public_url}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Ngrok connection warning: {e}")

    app.run(port=5001, debug=True, use_reloader=False, threaded=True)
