import json
import platform
import re
import sqlite3
import os
import uuid
import time
from flask import Flask, request, Response, stream_with_context, render_template
import requests
from pyngrok import ngrok
from bs4 import BeautifulSoup
import pypdf
import chromadb
from werkzeug.utils import secure_filename

app = Flask(__name__)

# CONFIGURATION
MODEL = "gemma4:e4b"
API_URL = "http://127.0.0.1:11434/api/chat"
UPLOAD_FOLDER = "uploads"
TOOLS_FOLDER = "tools"
NGROK_AUTH_TOKEN = "1xaBGSEtDnlLgIK663nvwSaOiRq_Vgj6aPE1FDxgpk9dh2MR" # Set your authtoken here
TAVILY_API_KEY = "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"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TOOLS_FOLDER, exist_ok=True)
CHARTS_FOLDER = "charts"
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# --- INSTRUCTION DATASET (Brain Manual) ---
INSTRUCTION_DATASET = {}
try:
    with open("instructions_dataset.json", "r", encoding="utf-8") as f:
        INSTRUCTION_DATASET = json.load(f)
    print("✅ Instruction dataset loaded successfully.")
except Exception as e:
    print(f"⚠️ Instruction dataset not found: {e}. Using hardcoded fallbacks.")

def get_route_examples(route_name, max_examples=5):
    """Pull few-shot examples from the instruction dataset for the router."""
    try:
        route_data = INSTRUCTION_DATASET.get("routes", {}).get(route_name, {})
        examples = route_data.get("examples", [])[:max_examples]
        return "\n".join([f'  User: "{ex["input"]}" => {ex["action"]}' for ex in examples])
    except:
        return ""

def get_global_rules():
    """Get anti-hallucination and loop prevention rules from the dataset."""
    try:
        rules = INSTRUCTION_DATASET.get("global_rules", {})
        anti_hall = rules.get("anti_hallucination_rules", [])
        loop_prev = rules.get("loop_prevention", [])
        combined = anti_hall + loop_prev
        return "\n".join([f"- {r}" for r in combined])
    except:
        return ""

def get_trigger_keywords(route_name):
    """Get trigger keywords for a specific route from the dataset."""
    try:
        return INSTRUCTION_DATASET.get("routes", {}).get(route_name, {}).get("trigger_keywords", [])
    except:
        return []

# --- CHROMA DB SETUP (LOCAL RAG) ---
try:
    chroma_client = chromadb.PersistentClient(path="memory_db_vectors")
    docs_collection = chroma_client.get_or_create_collection(name="local_docs")
    memory_collection = chroma_client.get_or_create_collection(name="long_term_memory")
except Exception as e:
    print(f"ChromaDB Init Error: {e}. Document search will be unavailable.")

# --- GRAPHRAG: SQLite Knowledge Graph ---
GRAPH_DB_PATH = "knowledge_graph.db"

def init_graph_db():
    conn = sqlite3.connect(GRAPH_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS kg_nodes
                 (id TEXT PRIMARY KEY, label TEXT, type TEXT, created_at REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS kg_edges
                 (id TEXT PRIMARY KEY, source TEXT, target TEXT, relation TEXT, created_at REAL)''')
    conn.commit()
    conn.close()

init_graph_db()

def graph_upsert_node(node_id, label, node_type):
    conn = sqlite3.connect(GRAPH_DB_PATH)
    conn.execute("INSERT OR IGNORE INTO kg_nodes VALUES (?, ?, ?, ?)",
                 (node_id, label[:200], node_type, time.time()))
    conn.commit()
    conn.close()

def graph_add_edge(source, target, relation="related_to"):
    edge_id = f"e_{uuid.uuid4().hex[:8]}"
    conn = sqlite3.connect(GRAPH_DB_PATH)
    conn.execute("INSERT OR IGNORE INTO kg_edges VALUES (?, ?, ?, ?, ?)",
                 (edge_id, source, target, relation, time.time()))
    conn.commit()
    conn.close()

def graph_get_linked_memories(entity_ids):
    """Given a list of entity node IDs, find all memory IDs linked to them."""
    if not entity_ids:
        return []
    conn = sqlite3.connect(GRAPH_DB_PATH)
    placeholders = ",".join(["?"]*len(entity_ids))
    rows = conn.execute(
        f"SELECT DISTINCT source FROM kg_edges WHERE target IN ({placeholders}) "
        f"AND source LIKE 'mem_%'",
        entity_ids
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

def graph_get_all_nodes():
    conn = sqlite3.connect(GRAPH_DB_PATH)
    nodes = conn.execute("SELECT id, label, type FROM kg_nodes").fetchall()
    conn.close()
    return nodes

def graph_get_all_edges():
    conn = sqlite3.connect(GRAPH_DB_PATH)
    edges = conn.execute("SELECT source, target, relation FROM kg_edges").fetchall()
    conn.close()
    return edges

def extract_entities(text, model_name=MODEL):
    """Uses Ollama to extract named entities from a text snippet. Returns list of strings."""
    try:
        prompt = (f"Extract all named entities (projects, names, tools, bugs, dates, companies, features) "
                  f"from this text. Return ONLY a comma-separated list, nothing else.\n\nText: {text[:500]}")
        r = requests.post(API_URL, json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=15)
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "").strip()
        entities = [e.strip().lower() for e in raw.split(",") if e.strip() and len(e.strip()) > 2]
        return entities[:10]  # Cap at 10 entities per memory
    except Exception:
        return []

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



def save_memory(fact, model_name=MODEL):
    try:
        mem_id = f"mem_{uuid.uuid4().hex[:8]}"
        memory_collection.add(
            documents=[fact.strip()],
            metadatas=[{"timestamp": time.time()}],
            ids=[mem_id]
        )
        # --- GRAPHRAG: Extract entities and wire into knowledge graph ---
        graph_upsert_node(mem_id, fact.strip()[:120], "memory")
        entities = extract_entities(fact, model_name)
        for ent in entities:
            ent_id = f"ent_{ent.replace(' ', '_')[:40]}"
            graph_upsert_node(ent_id, ent, "entity")
            graph_add_edge(mem_id, ent_id, "mentions")
        return True
    except Exception as e:
        print(f"Memory DB Error: {e}")
        return False

def retrieve_relevant_memories(query_text):
    try:
        if memory_collection.count() == 0:
            return []
        # Step 1: Vector search for top-5 direct matches
        results = memory_collection.query(query_texts=[query_text], n_results=5)
        direct_ids = results.get('ids', [[]])[0] if results else []
        direct_docs = results.get('documents', [[]])[0] if results else []

        # Step 2: GraphRAG — find entities in query, walk graph to find connected memories
        query_entities = extract_entities(query_text)
        entity_ids = [f"ent_{e.replace(' ', '_')[:40]}" for e in query_entities]
        graph_mem_ids = graph_get_linked_memories(entity_ids)

        # Step 3: Fetch graph-linked memory docs not already in direct results
        extra_docs = []
        if graph_mem_ids:
            new_ids = [i for i in graph_mem_ids if i not in direct_ids][:3]
            if new_ids:
                fetched = memory_collection.get(ids=new_ids, include=["documents"])
                extra_docs = fetched.get("documents", []) if fetched else []

        return direct_docs + extra_docs
    except Exception:
        return []

# --- WEB SEARCH & SCRAPING ---
from bs4 import BeautifulSoup
import urllib.parse
def search_web(query, tavily_key=""):
    if tavily_key and tavily_key.strip():
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key.strip(),
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
                "include_images": False,
                "include_answer": False
            }
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            res_strings = [f"- Source [URL: {r.get('url', '')}]: {r.get('title', '')}\n  Snippet: {r.get('content', '')}" for r in results]
            if res_strings:
                return "\n\n".join(res_strings)
        except Exception:
            pass # Fall back to DuckDuckGo

    try:
        url = "https://lite.duckduckgo.com/lite/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"q": query}
        r = requests.post(url, data=data, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        results = []
        for tr in soup.find_all('tr'):
            td = tr.find('td', class_='result-snippet')
            if td:
                a_tag = tr.previous_sibling.find('a', class_='result-url')
                if a_tag:
                    href = a_tag.get('href', '').strip()
                    if href.startswith('//duckduckgo.com/l/?uddg='):
                        href = urllib.parse.unquote(href.split('uddg=')[1].split('&')[0])
                    snippet = td.text.strip()
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
        import re
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) > 5000:
            text = text[:5000] + "... [Content Truncated]"
            
        return text if text else "No readable text found on page."
    except Exception as e:
        return f"Failed to read URL: {str(e)}"

def get_search_urls(query, tavily_key="", max_urls=3):
    """Returns list of (url, title, snippet) tuples from search results."""
    urls = []
    if tavily_key and tavily_key.strip():
        try:
            r = requests.post("https://api.tavily.com/search", json={
                "api_key": tavily_key.strip(),
                "query": query,
                "search_depth": "advanced",
                "max_results": max_urls + 2,
                "include_images": False,
                "include_answer": False
            }, timeout=15)
            r.raise_for_status()
            for item in r.json().get("results", [])[:max_urls]:
                u = item.get("url", "")
                if u:
                    urls.append((u, item.get("title", u), item.get("content", "")))
            if urls:
                return urls
        except Exception:
            pass
    # DuckDuckGo fallback
    try:
        r = requests.post("https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"},
            timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tr in soup.find_all('tr'):
            td = tr.find('td', class_='result-snippet')
            if td:
                a_tag = tr.previous_sibling.find('a', class_='result-url') if tr.previous_sibling else None
                if a_tag:
                    href = a_tag.get('href', '').strip()
                    if href.startswith('//duckduckgo.com/l/?uddg='):
                        href = urllib.parse.unquote(href.split('uddg=')[1].split('&')[0])
                    urls.append((href, href, td.text.strip()))
            if len(urls) >= max_urls:
                break
    except Exception:
        pass
    return urls

# --- SELF-ANALYSIS BRAIN (Repository Intelligence) ---
CODEBASE_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_EXTENSIONS = {".py", ".html", ".json", ".md", ".txt", ".js", ".css"}
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", "memory_db_vectors", "uploads", "tools"}

def scan_codebase(focus_query=""):
    """Scans the codebase and returns a structured snapshot for AI analysis."""
    try:
        file_tree = []
        file_contents = []
        total_lines = 0
        total_files = 0

        for dirpath, dirnames, filenames in os.walk(CODEBASE_ROOT):
            # Exclude unwanted dirs
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in CODE_EXTENSIONS:
                    continue
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, CODEBASE_ROOT)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    lines = content.count('\n')
                    total_lines += lines
                    total_files += 1
                    file_tree.append(f"  {rel_path} ({lines} lines, {len(content)} chars)")
                    # Only include content of primary code files to avoid token explosion
                    if ext in {".py", ".html"} and len(content) < 40000:
                        # Trim individual files to 6000 chars max
                        snippet = content[:6000] + ("\n... [Truncated]" if len(content) > 6000 else "")
                        file_contents.append(f"\n{'='*60}\nFILE: {rel_path}\n{'='*60}\n{snippet}")
                except Exception:
                    continue

        tree_str = "\n".join(file_tree)
        content_str = "\n".join(file_contents)

        return (
            f"[CODEBASE OVERVIEW]\n"
            f"Root: {CODEBASE_ROOT}\n"
            f"Total files scanned: {total_files} | Total lines: {total_lines}\n"
            f"\n[FILE TREE]\n{tree_str}\n"
            f"\n[FILE CONTENTS]\n{content_str}"
        )
    except Exception as e:
        return f"Codebase scan failed: {e}"


# --- LOCAL PYTHON EXECUTION ---
import sys, io, traceback

# Global, persistent namespace so repeated code blocks reuse imports and variables
persistent_python_env = {}

def execute_python(code_str):
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        exec(code_str, persistent_python_env)
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

# --- VISION MODEL ---
VISION_MODEL = "llava"  # Ollama vision model — run: ollama pull llava

# --- ROUTER NANO-MODEL ---
ROUTER_MODEL = "qwen2:0.5b"  # Ultra-fast routing model — run: ollama pull qwen2:0.5b

# --- AGENTIC LOOP AND PARSER ---
def ask_ai_stream(messages, target_model=MODEL, tools_enabled=True, router_enabled=True, force_web_search=False, thinking_enabled=False, tavily_key="", vision_enabled=False):
    # --- CONTEXT PRUNING (SLIDING WINDOW) ---
    MAX_MESSAGES = 10
    if len(messages) > MAX_MESSAGES:
        # Save any system messages (like injected doc context or initial prompts)
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        # Keep only the most recent N messages
        recent_msgs = [m for m in messages if m.get("role") != "system"][-MAX_MESSAGES:]
        messages = sys_msgs + recent_msgs

    # --- VISION DETECTION: Only activate if user toggled Vision ON and images are present ---
    has_images = vision_enabled and any(m.get("images") for m in messages)
    if has_images:
        def generate_vision():
            yield json.dumps({"type": "status", "text": "👁️ Vision Node activated — analyzing image..."}) + "\n"
            try:
                response = requests.post(
                    API_URL,
                    json={"model": VISION_MODEL, "messages": messages, "stream": True},
                    stream=True,
                    timeout=180
                )
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            chunk = data.get("message", {}).get("content", "") or data.get("response", "")
                            if chunk:
                                yield json.dumps({"type": "content", "text": chunk}) + "\n"
                        except:
                            continue
            except Exception as e:
                yield json.dumps({"type": "content", "text": f"Vision Error: {e}"}) + "\n"
        return generate_vision()

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
                tool_instructions = f"""You have tools. To use a tool, you MUST use the EXACT syntax below. Do NOT add conversational filler! You are a machine code generator. To use a tool, output ONLY the bracketed string.
- Search web: [SEARCH: query]
- Read url: [READ_URL: https://...]
- Save memory: [MEM_SAVE: fact]
- Search docs: [SEARCH_DOC: query]
- Run python: [PYTHON: print("hello")]
- Run shell: [RUN_SHELL: command]

Example: If asked to ping google, output exactly and ONLY:
[RUN_SHELL: ping google.com]"""
            else:
                user_msgs = [m["content"] for m in current_run_msgs if m["role"] == "user"]
                last_msgs = "\n".join([f"- {m}" for m in user_msgs[-3:]]) if user_msgs else "General query"
                
                # Check for "Current Event", "System", or "Documents" keywords to bias the router
                ds_research_kw = get_trigger_keywords("RESEARCH")
                ds_system_kw = get_trigger_keywords("SYSTEM")
                ds_docs_kw = get_trigger_keywords("DOCS")
                current_event_keywords = list(set(["current", "latest", "now", "who is", "today", "news", "status", "price", "population", "compare", "details of", "how many", "weather"] + ds_research_kw))
                system_keywords = list(set(["install", "pip", "modulenotfounderror", "missing module", "download"] + ds_system_kw))
                doc_keywords = list(set(["read the file", "pdf", "document", "uploaded", "resume", "my file", "the report"] + ds_docs_kw))
                last_user_msg = user_msgs[-1].lower() if user_msgs else ""
                
                data_keywords = ["csv", "excel", "xlsx", "spreadsheet", "dataset", "data analysis", "chart", "plot", "graph", "revenue", "sales", "statistics", "insights from", "visualize", "correlation", "monthly", "trend"]
                analysis_keywords = ["scan", "analyze", "analyse", "your code", "your codebase", "your own", "your files", "review your", "optimize yourself", "security flaw", "find bug", "redundant", "refactor", "self-analysis", "inspect"]
                nudge_research = any(kw in last_user_msg for kw in current_event_keywords)
                nudge_system = any(kw in last_user_msg for kw in system_keywords)
                nudge_docs = "has attached the following document" in last_user_msg or any(kw in last_user_msg for kw in doc_keywords)
                nudge_analysis = any(kw in last_user_msg for kw in analysis_keywords)
                nudge_data = any(kw in last_user_msg for kw in data_keywords)
                    
                # Build few-shot examples for the router from the instruction dataset
                research_examples = get_route_examples("RESEARCH", 3)
                code_examples = get_route_examples("CODE", 3)
                docs_examples = get_route_examples("DOCS", 2)
                system_examples = get_route_examples("SYSTEM", 2)
                general_examples = get_route_examples("GENERAL", 2)

                few_shot_block = ""
                if research_examples:
                    few_shot_block += f"\nExamples for RESEARCH:\n{research_examples}"
                if code_examples:
                    few_shot_block += f"\nExamples for CODE:\n{code_examples}"
                if docs_examples:
                    few_shot_block += f"\nExamples for DOCS:\n{docs_examples}"
                if system_examples:
                    few_shot_block += f"\nExamples for SYSTEM:\n{system_examples}"
                if general_examples:
                    few_shot_block += f"\nExamples for GENERAL:\n{general_examples}"
                    
                router_prompt = f"""You are a routing agent. Read the user's messages to understand the context.
Categorize the user's LATEST request into EXACTLY ONE of these strings:
[ROUTE: RESEARCH] - Use for web search, current events, news, identifying people/leaders, any factual lookup.
[ROUTE: CODE] - Use for math, logic, scripting, or python code.
[ROUTE: DOCS] - Use for searching or reading uploaded documents/PDFS.
[ROUTE: SYSTEM] - Use for executing local system shell commands (e.g. installing pip packages).
[ROUTE: ANALYSIS] - Use when the user asks to scan, review, optimize, or analyze the AI's own code/codebase.
[ROUTE: DATA] - Use when the user uploads or mentions a CSV/Excel file, asks for charts, plots, statistics, or data insights.
[ROUTE: GENERAL] - Use for general conversation only IF you are 100% sure the answer doesn't need live data.

IMPORTANT: If the user asks about anything CURRENT (leaders, dates, news, status), you MUST choose [ROUTE: RESEARCH].
IMPORTANT: If a Python script previously failed due to a missing module/library, or if the user asks to install/download something, you MUST choose [ROUTE: SYSTEM].
IMPORTANT: If the user uploaded a document and asks you to read or summarize it, you MUST choose [ROUTE: DOCS].
IMPORTANT: If the user asks to scan, review, or analyze the AI's own code or source files, you MUST choose [ROUTE: ANALYSIS].
IMPORTANT: If the user mentions CSV, Excel, charts, plots, data analysis, or statistics, you MUST choose [ROUTE: DATA].
{few_shot_block}

Recent User Messages Context:
{last_msgs}

Output ONLY the exact category string and nothing else."""

                if force_web_search or (nudge_research and not nudge_docs and not nudge_analysis):
                    route_text = "[ROUTE: RESEARCH]"
                elif nudge_system:
                    route_text = "[ROUTE: SYSTEM]"
                elif nudge_docs:
                    route_text = "[ROUTE: DOCS]"
                elif nudge_analysis:
                    route_text = "[ROUTE: ANALYSIS]"
                elif nudge_data:
                    route_text = "[ROUTE: DATA]"
                else:
                    try:
                        router_res = requests.post(API_URL, json={
                            "model": ROUTER_MODEL, 
                            "messages": [{"role": "user", "content": router_prompt}], 
                            "stream": False
                        }, timeout=10)
                        router_res.raise_for_status()
                        router_data = router_res.json()
                        route_text = router_data.get("message", {}).get("content", "").strip()
                    except Exception as e:
                        route_text = "[ROUTE: GENERAL]" # fallback
                    
                if "[ROUTE: RESEARCH]" in route_text:
                    cat = "RESEARCH"
                    icon = "🔍 Explorer Node"
                    tool_instructions = (
                        "You are the Explorer Node — a multi-step research agent.\n"
                        "For SIMPLE factual questions, use: [SEARCH: query]\n"
                        "For COMPLEX questions needing deep research (guides, comparisons, multi-topic reports), use: [DEEP_RESEARCH: query]\n"
                        "DO NOT CONVERSE. Output ONLY the tool syntax. DO NOT ANSWER FROM MEMORY."
                    )
                elif "[ROUTE: ANALYSIS]" in route_text:
                    cat = "ANALYSIS"
                    icon = "🧬 Analysis Node"
                    tool_instructions = (
                        "You are the Analysis Node — a self-aware code intelligence agent.\n"
                        "You have the power to scan your own source files and codebase.\n"
                        "To perform an analysis, output EXACTLY AND ONLY this syntax:\n"
                        "[ANALYZE_CODE: specific focus area or question]\n"
                        "Examples:\n"
                        "[ANALYZE_CODE: find security vulnerabilities in main.py]\n"
                        "[ANALYZE_CODE: suggest routing logic optimizations]\n"
                        "[ANALYZE_CODE: identify redundant code across all files]\n"
                        "DO NOT CONVERSE. DO NOT GUESS. Output only the bracketed tool syntax."
                    )
                elif "[ROUTE: CODE]" in route_text:
                    cat = "CODE"
                    icon = "🤖 Coder Node"
                    tool_instructions = (
                        "You are the Coder Node. Do NOT converse.\n"
                        "To calculate or execute code, output EXACTLY AND ONLY:\n[PYTHON: print('hello')]\n"
                        "To write a file to disk, output EXACTLY AND ONLY:\n[WRITE_FILE: path/to/file.ext]\ncontent\n[/WRITE_FILE]\n"
                        "To read a file from disk, output EXACTLY AND ONLY:\n[READ_FILE: path/to/file.ext]\n"
                    )
                elif "[ROUTE: DATA]" in route_text:
                    cat = "DATA"
                    icon = "📊 Data Agent"
                    tool_instructions = (
                        "You are the Data Analysis Agent. You can analyze CSV/Excel files the user has uploaded.\n"
                        "To analyze data, output EXACTLY AND ONLY this syntax:\n"
                        "[ANALYZE_DATA: description of what analysis and chart to generate]\n"
                        "Examples:\n"
                        "[ANALYZE_DATA: show revenue by month as a bar chart]\n"
                        "[ANALYZE_DATA: generate a correlation heatmap of all numeric columns]\n"
                        "[ANALYZE_DATA: show top 10 rows and summary statistics]\n"
                        "DO NOT CONVERSE. Output only the bracketed syntax."
                    )
                elif "[ROUTE: DOCS]" in route_text:
                    cat = "DOCS"
                    icon = "📚 Document Node"
                    tool_instructions = "You are the Document Node. YOU HAVE A POWERFUL NATIVE DOCUMENT READER. IGNORE ALL PAST ERRORS. Do NOT rely on Python scripts to read files. To search and read documents, output EXACTLY AND ONLY this syntax:\n[SEARCH_DOC: specific query]\nDO NOT CONVERSE. DO NOT APOLOGIZE. JUST USE THE TOOL."
                elif "[ROUTE: SYSTEM]" in route_text:
                    cat = "SYSTEM"
                    icon = "💻 System Node"
                    os_type = "powershell" if platform.system() == "Windows" else "bash"
                    tool_instructions = f"You are the System Node. Do NOT converse. To control the OS ({os_type}), output EXACTLY AND ONLY this syntax:\n[RUN_SHELL: command name]\nExample:\n[RUN_SHELL: ping google.com]"
                else:
                    cat = "GENERAL"
                    icon = "🧠 General Node"
                    tool_instructions = "You are the General Node. If the user asks you to remember, save, or note down a fact, you MUST output ONLY this syntax:\n[MEM_SAVE: fact here]\nDo NOT converse if saving memory. Otherwise, converse normally. If you realize you lack factual data to answer a query definitively, self-correct and output EXACTLY AND ONLY:\n[SEARCH: specific query]"

            # Inject custom tools from file system
            available_tools_list = []
            if os.path.exists(TOOLS_FOLDER):
                for f in os.listdir(TOOLS_FOLDER):
                    if f.endswith('.py'):
                        available_tools_list.append(f)
            tools_str = ", ".join(available_tools_list) if available_tools_list else "None yet."
            
            tool_instructions += f"\n\n[SELF-EVOLVING TOOLS]\nYou can create persistent python scripts in the tools/ directory.\nTo create a tool, output EXACTLY AND ONLY:\n[SAVE_TOOL: filename.py]\n<python code here>\n[/SAVE_TOOL]\n\nTo execute an existing tool, use: [RUN_SHELL: python tools/filename.py --args]\nAvailable custom tools in tools/ folder: {tools_str}"

            # Design & Formatting Protocol
            formatting_protocol = "\n\n[OUTPUT DESIGN PROTOCOL]\nWhen giving conversational answers, you MUST look premium and professional. Rule 1: Use bolding for key terms. Rule 2: Use Markdown tables for comparisons or structured data. Rule 3: Use bullet points for lists. Rule 4: Do not output walls of text; use short punchy paragraphs. Rule 5: Be decisive and authoritative; remove filler phrases."
            tool_instructions += formatting_protocol

            # Inject global anti-hallucination and loop prevention rules from dataset
            global_rules = get_global_rules()
            if global_rules:
                tool_instructions += f"\n\n[CRITICAL RULES]\n{global_rules}"

            # Inject dynamic sub-agent instructions safely into system prompt
            if len(current_run_msgs) > 0 and current_run_msgs[0]["role"] == "system":
                current_run_msgs[0]["content"] += "\n\n" + tool_instructions
            else:
                current_run_msgs.insert(0, {"role": "system", "content": tool_instructions})
        
        if thinking_enabled:
            think_prompt = "You must deeply think about the problem before answering. Provide your step-by-step thinking process enclosed in <think>...</think> tags at the very beginning of your response."
            if len(current_run_msgs) > 0 and current_run_msgs[0]["role"] == "system":
                current_run_msgs[0]["content"] += "\n\n" + think_prompt
            else:
                current_run_msgs.insert(0, {"role": "system", "content": think_prompt})
        
        loop_count = 0
        max_loops = 5

        while loop_count < max_loops:
            buffer = ""
            full_response = ""
            tool_triggered = False
            
            for chunk in ollama_stream(current_run_msgs):
                full_response += chunk
                buffer += chunk

                if '[' in buffer:
                    analyze_data_match = re.search(r'\[ANALYZE_DATA:\s*(.*?)\]', buffer)
                    analyze_match = re.search(r'\[ANALYZE_CODE:\s*(.*?)\]', buffer)
                    deep_research_match = re.search(r'\[DEEP_RESEARCH:\s*(.*?)\]', buffer)
                    search_match = re.search(r'\[SEARCH:\s*(.*?)\]', buffer)
                    mem_save_match = re.search(r'\[MEM_SAVE:\s*(.*?)\]', buffer)
                    doc_search_match = re.search(r'\[SEARCH_DOC:\s*(.*?)\]', buffer)
                    read_url_match = re.search(r'\[READ_URL:\s*(.*?)\]', buffer)
                    python_match = re.search(r'\[PYTHON:\s*(.*?)\n?\]', buffer, re.DOTALL)
                    shell_match = re.search(r'\[RUN_SHELL:\s*(.*?)\n?\]', buffer, re.DOTALL)
                    save_tool_match = re.search(r'\[SAVE_TOOL:\s*([^\]]+)\](.*?)\[/SAVE_TOOL\]', buffer, re.DOTALL)
                    write_file_match = re.search(r'\[WRITE_FILE:\s*([^\]]+)\](.*?)\[/WRITE_FILE\]', buffer, re.DOTALL)
                    read_file_match = re.search(r'\[READ_FILE:\s*([^\]]+)\]', buffer)

                    if analyze_data_match:
                        instructions = analyze_data_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"📊 Data Agent activated: '{instructions}'"}) + "\n"
                        # Find the latest uploaded CSV/XLSX
                        data_file = None
                        for fname in sorted(os.listdir(UPLOAD_FOLDER), reverse=True):
                            if fname.endswith(('.csv', '.xlsx', '.xls')):
                                data_file = os.path.join(UPLOAD_FOLDER, fname)
                                break
                        if not data_file:
                            result_text = "No CSV or Excel file found. Please upload a data file first."
                        else:
                            try:
                                import pandas as pd
                                import matplotlib
                                matplotlib.use('Agg')
                                import matplotlib.pyplot as plt
                                import io as _io

                                df = pd.read_csv(data_file) if data_file.endswith('.csv') else pd.read_excel(data_file)
                                yield json.dumps({"type": "status", "text": f"📂 Loaded: {os.path.basename(data_file)} ({df.shape[0]} rows, {df.shape[1]} cols)"}) + "\n"

                                # Build a text summary for the AI
                                summary = f"File: {os.path.basename(data_file)}\nShape: {df.shape}\nColumns: {list(df.columns)}\n\nData Types:\n{df.dtypes.to_string()}\n\nSummary Stats:\n{df.describe().to_string()}\n\nFirst 5 rows:\n{df.head().to_string()}"

                                # Generate a chart with matplotlib
                                chart_name = f"chart_{uuid.uuid4().hex[:8]}.png"
                                chart_path = os.path.join(CHARTS_FOLDER, chart_name)
                                plt.style.use('dark_background')
                                fig, ax = plt.subplots(figsize=(10, 5))
                                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                                if numeric_cols:
                                    df[numeric_cols[:3]].plot(ax=ax, kind='bar' if len(df) <= 30 else 'line')
                                    ax.set_title(instructions[:60], color='white', fontsize=12, pad=10)
                                    ax.set_facecolor('#1a1a2e')
                                    fig.patch.set_facecolor('#0c0c0f')
                                    ax.tick_params(colors='#a1a1aa')
                                    plt.tight_layout()
                                    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
                                    plt.close(fig)
                                    chart_url = f"/charts/{chart_name}"
                                    yield json.dumps({"type": "chart", "url": chart_url}) + "\n"
                                    yield json.dumps({"type": "status", "text": "✅ Chart generated."}) + "\n"
                                else:
                                    chart_url = None

                                result_text = summary
                            except Exception as e:
                                result_text = f"Data analysis failed: {e}"

                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Data Analysis Results:\n{result_text}\n\nProvide a comprehensive insight report based on this data. Use markdown tables, bullet points, and highlight key findings."})
                        tool_triggered = True
                        break

                    elif analyze_match:
                        focus = analyze_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🧬 Analysis Node activated: '{focus}'"}) + "\n"
                        yield json.dumps({"type": "status", "text": "🗂️ Scanning codebase files..."}) + "\n"
                        codebase_snapshot = scan_codebase(focus)
                        yield json.dumps({"type": "status", "text": "✅ Codebase scanned. Generating analysis..."}) + "\n"
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": (
                            f"System Notice: Codebase Snapshot Ready. Focus: '{focus}'\n\n"
                            f"{codebase_snapshot}\n\n"
                            "Based on the above code snapshot, provide a detailed, structured analysis for the user's focus area. "
                            "Use headers, bullet points, and code blocks. Be specific with file names and line references."
                        )})
                        tool_triggered = True
                        break

                    elif deep_research_match:
                        query = deep_research_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🌐 Deep Research initiated: '{query}'"}) + "\n"
                        source_urls = get_search_urls(query, tavily_key, max_urls=3)
                        if not source_urls:
                            tool_result = "Deep research yielded no sources. Please try a different query."
                        else:
                            yield json.dumps({"type": "status", "text": f"📡 Found {len(source_urls)} sources. Reading each one..."}) + "\n"
                            all_content = []
                            for i, (src_url, title, snippet) in enumerate(source_urls):
                                yield json.dumps({"type": "status", "text": f"📖 Reading source {i+1}/{len(source_urls)}: {title[:60]}"}) + "\n"
                                page_text = read_url(src_url)
                                all_content.append(f"=== Source {i+1}: {title} ===\nURL: {src_url}\n{page_text}")
                            tool_result = "\n\n".join(all_content)
                            yield json.dumps({"type": "status", "text": f"✅ Deep Research complete. Synthesizing answer from {len(source_urls)} sources..."}) + "\n"

                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Deep Research Results (from {len(source_urls)} live sources):\n{tool_result}\n\nNow synthesize this data into a comprehensive, well-structured answer for the user's original query. Use headers, bullet points, and tables where appropriate."})
                        tool_triggered = True
                        break

                    elif search_match:
                        query = search_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🔍 ... searching on web for: '{query}'"}) + "\n"
                        tool_result = search_web(query, tavily_key)
                        yield json.dumps({"type": "status", "text": f"✅ Web search completed."}) + "\n"
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Web Search Results:\n{tool_result}\n\nUse this information to answer the initial query."})
                        tool_triggered = True
                        break
                        
                    elif mem_save_match:
                        fact = mem_save_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🧠 Saving memory..."}) + "\n"
                        save_memory(fact, target_model)
                        yield json.dumps({"type": "status", "text": f"✅ Memory saved."}) + "\n"
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Memory '{fact}' saved successfully. Acknowledge this."})
                        tool_triggered = True
                        break

                    elif doc_search_match:
                        query = doc_search_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"📚 Searching local documents for: {query}"}) + "\n"
                        tool_result = search_docs(query)
                        yield json.dumps({"type": "status", "text": f"✅ Document search completed."}) + "\n"
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Local Document Results:\n{tool_result}\n\nSynthesize this information to answer the initial query."})
                        tool_triggered = True
                        break

                    elif read_url_match:
                        url = read_url_match.group(1).strip()
                        yield json.dumps({"type": "status", "text": f"🌐 Reading webpage: {url}"}) + "\n"
                        tool_result = read_url(url)
                        yield json.dumps({"type": "status", "text": f"✅ Webpage read successfully."}) + "\n"
                        
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
                        
                        if "Python Execution Error:" in tool_result or "Traceback" in tool_result or "Error:" in tool_result:
                            yield json.dumps({"type": "status", "text": f"⚠️ Execution failed. Self-correcting..."}) + "\n"
                            current_run_msgs.append({"role": "user", "content": f"System Notice: Python Execution FAILED with error:\n{tool_result}\n\nDO NOT APOLOGIZE. Identify the bug and instantly write a fully corrected version of the python script using [PYTHON: ...]. You must fix this."})
                        else:
                            yield json.dumps({"type": "status", "text": f"✅ Execution finished."}) + "\n"
                            current_run_msgs.append({"role": "user", "content": f"System Notice: Python Execution Output (stdout/stderr):\n{tool_result}\n\nUse this information to answer the initial query."})
                        
                        tool_triggered = True
                        break

                    elif shell_match:
                        code = shell_match.group(1).strip()
                        yield json.dumps({"type": "action_request", "action": "RUN_SHELL", "command": code}) + "\n"
                        # Terminate the stream explicitly. The frontend will pick up execution.
                        break

                    elif save_tool_match:
                        filename = save_tool_match.group(1).strip()
                        code = save_tool_match.group(2).strip()
                        if code.startswith('```python'): code = code[9:]
                        elif code.startswith('```'): code = code[3:]
                        if code.endswith('```'): code = code[:-3]
                        
                        tool_path = os.path.join(TOOLS_FOLDER, filename)
                        with open(tool_path, 'w', encoding='utf-8') as f:
                            f.write(code.strip())
                            
                        yield json.dumps({"type": "status", "text": f"🛠️ Tool created & saved: {filename}"}) + "\n"
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: Custom tool successfully saved to {tool_path}."})
                        tool_triggered = True
                        break

                    elif write_file_match:
                        filepath = write_file_match.group(1).strip()
                        content = write_file_match.group(2).strip()
                        if content.startswith('```'): 
                            content = re.sub(r'^```[a-zA-Z]*\n', '', content)
                            if content.endswith('```'): content = content[:-3]
                        
                        try:
                            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(content.strip("\n"))
                            yield json.dumps({"type": "status", "text": f"💾 File created: {filepath}"}) + "\n"
                            tool_result = f"Successfully wrote to {filepath}"
                        except Exception as e:
                            yield json.dumps({"type": "status", "text": f"⚠️ Failed to write file: {e}"}) + "\n"
                            tool_result = f"Error writing file: {e}"
                            
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: [WRITE_FILE] Result:\n{tool_result}\n\nConform your next output according to the instructions."})
                        tool_triggered = True
                        break

                    elif read_file_match:
                        filepath = read_file_match.group(1).strip()
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            yield json.dumps({"type": "status", "text": f"📖 File read: {filepath}"}) + "\n"
                            tool_result = f"File Content for {filepath}:\n```\n{content}\n```"
                        except Exception as e:
                            yield json.dumps({"type": "status", "text": f"⚠️ Failed to read file: {e}"}) + "\n"
                            tool_result = f"Error reading file: {e}"
                        
                        current_run_msgs.append({"role": "assistant", "content": full_response})
                        current_run_msgs.append({"role": "user", "content": f"System Notice: [READ_FILE] Result:\n{tool_result}\n\nUse this information."})
                        tool_triggered = True
                        break
                    
                    if len(buffer) > 60:
                        if re.match(r'^\[(?:SEARCH|MEM_SAVE|SEARCH_DOC|READ_URL|PYTHON|RUN_SHELL|SAVE_TOOL|WRITE_FILE|READ_FILE):', buffer):
                            if len(buffer) > 20000: # safety bailout expanded for tools
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
                    clean_buf = re.sub(r'\[(?:SEARCH|MEM_SAVE|SEARCH_DOC|READ_URL|PYTHON|RUN_SHELL|SAVE_TOOL|WRITE_FILE|READ_FILE).*$', '', buffer, flags=re.DOTALL)
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
    thinking_enabled = data.get("thinking_enabled", False)
    vision_enabled = data.get("vision_enabled", False)
    tavily_key = data.get("tavily_api_key", "") or TAVILY_API_KEY
    
    return Response(stream_with_context(ask_ai_stream(messages, target_model, tools_enabled, router_enabled, force_web_search, thinking_enabled, tavily_key, vision_enabled)), mimetype='application/x-ndjson')

@app.route("/charts/<filename>")
def serve_chart(filename):
    from flask import send_from_directory
    return send_from_directory(CHARTS_FOLDER, filename)

@app.route("/memories_graph", methods=["GET"])
def get_memories_graph():
    try:
        # Core node
        nodes = [{"id": "nexus_core", "name": "Nexus Core", "val": 20, "color": "#14b8a6", "type": "core"}]
        links = []

        # Load all graph nodes (memories + entities)
        all_kg_nodes = graph_get_all_nodes()   # (id, label, type)
        all_kg_edges = graph_get_all_edges()   # (source, target, relation)

        known_ids = {"nexus_core"}
        for (node_id, label, ntype) in all_kg_nodes:
            if ntype == "memory":
                color = "#f59e0b"   # amber — memory nodes
                val = 8
            else:
                color = "#a855f7"   # violet — entity nodes
                val = 5
            nodes.append({
                "id": node_id,
                "name": label[:60],
                "val": val,
                "color": color,
                "type": ntype,
                "desc": f"[{ntype.upper()}] {label}"
            })
            known_ids.add(node_id)
            # Connect memory nodes to core
            if ntype == "memory":
                links.append({"source": "nexus_core", "target": node_id})

        # Add graph edges (memory → entity relationships)
        for (source, target, relation) in all_kg_edges:
            if source in known_ids and target in known_ids:
                links.append({"source": source, "target": target, "label": relation})

        # Merge legacy vector DB memories that might not be in the Graph yet
        if memory_collection.count() > 0:
            data = memory_collection.get(include=["documents"])
            for doc, mid in zip(data.get("documents", []), data.get("ids", [])):
                if mid not in known_ids:
                    nodes.append({"id": mid, "name": doc[:60], "val": 6, "color": "#f59e0b", "type": "memory", "desc": f"[MEMORY] {doc}"})
                    links.append({"source": "nexus_core", "target": mid})
                    known_ids.add(mid)

        return {"nodes": nodes, "links": links}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    try:
        public_url = ngrok.connect(5001).public_url
        print("\n" + "="*80)
        print(f"🚀 Nexus Engine is LIVE on the internet at: {public_url}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Ngrok connection warning: {e}")

    app.run(port=5001, debug=True, use_reloader=False, threaded=True)
