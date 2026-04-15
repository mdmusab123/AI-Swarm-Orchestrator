import json
from flask import Flask, request, Response, stream_with_context, render_template_string
import requests
import subprocess
import time
import platform

app = Flask(__name__)

# CONFIGURATION
MODEL = "gemma4:e2b"
# Ensure this URL is accessible and pointing to your Ollama/AI provider
API_URL = "http://127.0.0.1:11434/api/chat"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Assistant</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Markdown & Security -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js"></script>
    
    <!-- Highlight.js for Code Blocks -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

    <style>
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #424242; border-radius: 10px; }
        
        .prose pre { background-color: #171717 !important; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; border: 1px solid #333; }
        .prose code { background-color: #333; padding: 0.1rem 0.3rem; border-radius: 0.25rem; font-family: monospace; }
        .prose pre code { background-color: transparent; padding: 0; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        .typing-dot { animation: typing 1.4s infinite ease-in-out both; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        body { background-color: #212121; }
        textarea { resize: none; transition: height 0.1s ease; }
    </style>
</head>

<body class="text-gray-200 h-screen flex overflow-hidden font-sans antialiased bg-[#212121]">

    <!-- Sidebar (Desktop) -->
    <div class="hidden md:flex w-[260px] bg-[#171717] flex-col h-full border-r border-[#333]">
        <div class="p-4">
            <button onclick="startNewChat()" class="flex items-center gap-2 w-full p-3 rounded-lg hover:bg-[#2f2f2f] transition-colors border border-[#424242] text-sm text-gray-300">
                <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                New Chat
            </button>
        </div>
        <div class="flex-1 overflow-y-auto p-4 pt-0">
            <p class="text-xs font-semibold text-gray-500 mb-3 px-2 uppercase tracking-wider">History</p>
            <div id="chat-list" class="flex flex-col space-y-1"></div>
        </div>
        <div class="p-4 border-t border-[#333]">
            <button onclick="toggleSettings()" class="flex items-center gap-3 w-full p-2 rounded-lg hover:bg-[#2f2f2f] transition-colors text-sm text-gray-300">
                <div class="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold">U</div>
                Settings
            </button>
        </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 flex flex-col h-full relative">
        <div id="chat-container" class="flex-1 overflow-y-auto w-full pb-40 pt-8">
            <div id="chat" class="max-w-3xl mx-auto flex flex-col space-y-8 px-4">
                <!-- Messages populate here -->
            </div>
        </div>

        <!-- Input Area -->
        <div class="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#212121] via-[#212121] to-transparent pt-10 pb-8 px-4">
            <div class="max-w-3xl mx-auto relative">
                <div class="bg-[#2f2f2f] rounded-2xl border border-[#424242] focus-within:border-gray-400 transition-all flex items-end p-2 shadow-2xl">
                    <textarea 
                        id="input" 
                        rows="1"
                        placeholder="Message AI Assistant..." 
                        class="w-full max-h-[200px] bg-transparent text-gray-100 placeholder-gray-500 px-3 py-3 border-none outline-none focus:ring-0 text-md"
                    ></textarea>
                    
                    <button id="send-btn" onclick="sendMessage()" class="p-2.5 mb-1 mr-1 bg-white text-black hover:bg-gray-200 rounded-xl transition-all disabled:opacity-30 disabled:cursor-not-allowed">
                        <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div id="settings-modal" class="hidden fixed inset-0 bg-black/80 z-50 flex items-center justify-center backdrop-blur-sm">
        <div class="bg-[#171717] border border-[#333] rounded-2xl w-full max-w-lg p-6 shadow-2xl">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-xl font-semibold">System Settings</h2>
                <button onclick="toggleSettings()" class="text-gray-400 hover:text-white">
                    <svg class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                </button>
            </div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-400 mb-2">System Persona</label>
                    <textarea id="sys-prompt" rows="4" class="w-full bg-[#212121] text-gray-200 border border-[#424242] rounded-lg p-3 focus:border-white focus:outline-none" placeholder="e.g. You are a helpful assistant..."></textarea>
                </div>
                <button onclick="saveSettings()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors">Save Changes</button>
            </div>
        </div>
    </div>

    <script>
        // Use standard variable declarations to avoid scope issues
        let chats = [];
        let currentChatId = null;
        let messageHistory = [];
        let systemPrompt = localStorage.getItem('ai_sys_prompt') || '';

        const chatEl = document.getElementById("chat");
        const chatContainer = document.getElementById("chat-container");
        const inputEl = document.getElementById("input");
        const sendBtn = document.getElementById("send-btn");
        const chatListEl = document.getElementById("chat-list");

        // Initialization
        document.addEventListener("DOMContentLoaded", () => {
            const stored = localStorage.getItem('ai_chats_v2');
            chats = stored ? JSON.parse(stored) : [];
            document.getElementById("sys-prompt").value = systemPrompt;
            
            if (chats.length === 0) {
                startNewChat();
            } else {
                loadChat(chats[0].id);
            }
            renderSidebar();
        });

        // Event: Auto-resize textarea
        inputEl.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight > 200 ? 200 : this.scrollHeight) + 'px';
        });

        // Event: Enter to send
        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function toggleSettings() {
            document.getElementById("settings-modal").classList.toggle('hidden');
        }

        function saveSettings() {
            systemPrompt = document.getElementById("sys-prompt").value;
            localStorage.setItem('ai_sys_prompt', systemPrompt);
            toggleSettings();
        }

        function startNewChat() {
            currentChatId = Date.now().toString();
            messageHistory = [];
            chats.unshift({ id: currentChatId, title: 'New Chat', messages: [] });
            saveState();
            loadChat(currentChatId);
        }

        function loadChat(id) {
            currentChatId = id;
            const chat = chats.find(c => c.id === id);
            messageHistory = chat ? [...chat.messages] : [];
            chatEl.innerHTML = "";
            
            if (messageHistory.length === 0) {
                appendMessage("ai", "Hello! I'm your AI Assistant. How can I help you today?");
            } else {
                messageHistory.forEach(msg => appendMessage(msg.role, msg.content));
            }
            renderSidebar();
        }

        function renderSidebar() {
            chatListEl.innerHTML = '';
            chats.forEach(chat => {
                const item = document.createElement('div');
                const activeClass = chat.id === currentChatId ? 'bg-[#2f2f2f] text-white' : 'hover:bg-[#262626] text-gray-400';
                item.className = `p-2.5 rounded-lg text-sm cursor-pointer truncate transition-all ${activeClass}`;
                item.innerText = chat.title;
                item.onclick = () => loadChat(chat.id);
                chatListEl.appendChild(item);
            });
        }

        function appendMessage(role, content, id = null) {
            const div = document.createElement("div");
            div.className = "flex gap-4 w-full group";
            if (id) div.id = id;

            const isAI = role === 'ai';
            const avatar = isAI 
                ? `<div class="w-8 h-8 rounded-full bg-white flex items-center justify-center shrink-0"><svg class="w-5 h-5 text-black" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8V4m0 0L9 7m3-3l3 3m-9 5h12m-6 4v4m0 0l-3-3m3 3l3-3"></path></svg></div>`
                : `<div class="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shrink-0 text-xs font-bold">U</div>`;

            div.innerHTML = `
                ${avatar}
                <div class="flex-1 min-w-0">
                    <div class="prose prose-invert max-w-none text-[15px] leading-relaxed content-area">
                        ${isAI && content !== '...' ? formatMarkdown(content) : content}
                    </div>
                </div>
            `;

            if (content === '...') {
                div.querySelector('.content-area').innerHTML = `
                    <div class="flex space-x-1 py-2">
                        <div class="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
                        <div class="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
                        <div class="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
                    </div>
                `;
            }

            chatEl.appendChild(div);
            scrollToBottom();
            return div;
        }

        function formatMarkdown(text) {
            if (typeof marked === 'undefined') return text;
            const html = marked.parse(text, { breaks: true });
            return DOMPurify.sanitize(html);
        }

        function scrollToBottom() {
            chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
        }

        async function sendMessage() {
            const text = inputEl.value.trim();
            if (!text || sendBtn.disabled) return;

            // Update UI
            inputEl.value = "";
            inputEl.style.height = "auto";
            inputEl.disabled = true;
            sendBtn.disabled = true;

            appendMessage("user", text);
            messageHistory.push({ role: "user", content: text });

            // Temporary loading message
            const loadingId = "loading-" + Date.now();
            const aiMsgDiv = appendMessage("ai", "...", loadingId);
            const contentArea = aiMsgDiv.querySelector('.content-area');

            try {
                let fullHistory = [];
                if (systemPrompt) fullHistory.push({ role: "system", content: systemPrompt });
                fullHistory = [...fullHistory, ...messageHistory];

                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ messages: fullHistory })
                });

                if (!response.ok) throw new Error("Server Error");

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let accumulated = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    accumulated += decoder.decode(value);
                    contentArea.innerHTML = formatMarkdown(accumulated);
                    scrollToBottom();
                }

                messageHistory.push({ role: "assistant", content: accumulated });
                
                // Update chat title if it's the first message
                const currentChat = chats.find(c => c.id === currentChatId);
                if (currentChat && currentChat.title === 'New Chat') {
                    currentChat.title = text.substring(0, 30) + (text.length > 30 ? '...' : '');
                }
                if (currentChat) currentChat.messages = [...messageHistory];
                
                saveState();
                renderSidebar();

            } catch (err) {
                contentArea.innerHTML = `<span class="text-red-400 font-medium italic">Error: Connection lost.</span>`;
            } finally {
                inputEl.disabled = false;
                sendBtn.disabled = false;
                inputEl.focus();
                hljs.highlightAll();
            }
        }

        function saveState() {
            localStorage.setItem('ai_chats_v2', JSON.stringify(chats));
        }
    </script>
</body>
</html>
"""

def ask_ai_stream(messages):
    try:
        response = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True 
            },
            stream=True,       
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    data = json.loads(decoded_line)
                    # Check Ollama's common response structures
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                    elif "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        yield f" [Backend Error: {str(e)}] "

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    return Response(stream_with_context(ask_ai_stream(messages)), mimetype='text/plain')

def install_cloudflared():
    system = platform.system().lower()
    if system == 'linux':
        try:
            subprocess.run(['curl', '-L', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', '-o', 'cloudflared'], check=True)
            subprocess.run(['chmod', '+x', 'cloudflared'], check=True)
            try:
                subprocess.run(['sudo', 'mv', 'cloudflared', '/usr/local/bin/'], check=True)
            except subprocess.CalledProcessError:
                print("Installed cloudflared locally. Ensure it's in your PATH or run from the script directory.")
        except subprocess.CalledProcessError:
            print("Failed to install cloudflared. Please install manually.")
    else:
        print("This script is configured for Linux only. Please install cloudflared manually from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/")

if __name__ == "__main__":
    app.run(port=5020, debug=True)
