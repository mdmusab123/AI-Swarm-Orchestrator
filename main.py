import json
import re
import sqlite3
from flask import Flask, request, Response, stream_with_context, render_template
import requests
from pyngrok import ngrok
from duckduckgo_search import DDGS

app = Flask(__name__)

# CONFIGURATION
MODEL = "gemma4:e2b"
API_URL = "http://127.0.0.1:11434/api/chat"

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
                return [row[0] for row in rows[-20:]] # Return up to 20 recent facts
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
    # Inject memory context explicitly for the system prompt
    if tools_enabled:
        memories = get_memories()
        if memories:
            memo_block = "User's saved long-term memories:\n" + "\n".join([f"- {m}" for m in memories])
            messages.insert(0, {"role": "system", "content": memo_block})

    def ollama_stream(msgs):
        try:
            response = requests.post(
                API_URL,
                json={
                    "model": MODEL,
                    "messages": msgs,
                    "stream": True 
                },
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
        max_loops = 3 # Anti-infinite loop mechanism

        while loop_count < max_loops:
            buffer = ""
            full_response = ""
            tool_triggered = False
            
            for chunk in ollama_stream(current_run_msgs):
                full_response += chunk
                buffer += chunk

                # Sniff for Tool Tags `[SEARCH:` or `[MEM_SAVE:`
                if '[' in buffer:
                    search_match = re.search(r'\[SEARCH:\s*(.*?)\]', buffer)
                    mem_save_match = re.search(r'\[MEM_SAVE:\s*(.*?)\]', buffer)
                    
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
                    
                    # Prevent buffer from growing infinitely and hanging the UI stream
                    if len(buffer) > 60:
                        idx = buffer.rfind('[')
                        if idx > 0:
                            # Flush safe content before the '['
                            yield json.dumps({"type": "content", "text": buffer[:idx]}) + "\n"
                            buffer = buffer[idx:]
                        else:
                            # '[' is at index 0 but it's too long, likely not a tool flag
                            yield json.dumps({"type": "content", "text": buffer}) + "\n"
                            buffer = ""
                else:
                    # Safe to emit
                    yield json.dumps({"type": "content", "text": buffer}) + "\n"
                    buffer = ""

            if tool_triggered and loop_count < max_loops - 1:
                loop_count += 1
                continue
            else:
                # Flush the remains
                if buffer:
                    # Strip any partial incomplete tags that got left at the end
                    clean_buf = re.sub(r'\[(?:SEARCH|MEM_SAVE).*$', '', buffer)
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
    
    # We must format the response as ndjson so the frontend can display partial updates gracefully
    return Response(stream_with_context(ask_ai_stream(messages, tools_enabled)), mimetype='application/x-ndjson')

if __name__ == "__main__":
    # ngrok.set_auth_token("1xaBGSEtDnlLgIK663nvwSaOiRq_Vgj6aPE1FDxgpk9dh2MR")
    
    try:
        public_url = ngrok.connect(5001).public_url
        print("\n" + "="*80)
        print(f"🚀 Nexus Engine is LIVE on the internet at: {public_url}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Ngrok connection warning: {e}")

    app.run(port=5001, debug=True, use_reloader=False, threaded=True)
