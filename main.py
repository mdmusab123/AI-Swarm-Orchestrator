import json
from flask import Flask, request, Response, stream_with_context, render_template_string
import requests

app = Flask(__name__)

MODEL = "phi3:mini"
API_URL = "http://127.0.0.1:11434/api/chat"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Assistant</title>
    
    <!-- Tailwind CSS for styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    
    <!-- DOMPurify for sanitizing HTML -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js"></script>
    
    <!-- Highlight.js for code syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>

    <style>
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #424242; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #525252; }

        /* Typography & Markdown Styles */
        .prose pre { 
            background-color: #171717 !important; 
            padding: 1rem; 
            overflow-x: auto; 
        }
        .prose code { 
            background-color: #2f2f2f; 
            padding: 0.2rem 0.4rem; 
            border-radius: 0.25rem; 
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.875em; 
        }
        .prose pre code { background-color: transparent; padding: 0; }
        .prose p { margin-bottom: 1em; line-height: 1.6; }
        .prose p:last-child { margin-bottom: 0; }
        .prose ul { list-style-type: disc; padding-left: 1.5em; margin-bottom: 1em;}
        .prose ol { list-style-type: decimal; padding-left: 1.5em; margin-bottom: 1em;}
        .prose strong { color: #f3f4f6; font-weight: 600; }
        .prose a { color: #60a5fa; text-decoration: underline; }
        .prose table { width: 100%; border-collapse: collapse; margin-bottom: 1em; }
        .prose th, .prose td { border: 1px solid #424242; padding: 0.5rem; text-align: left; }
        .prose th { background-color: #2f2f2f; }
        
        /* Typing animation */
        .typing-dot { animation: typing 1.4s infinite ease-in-out both; }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        body { background-color: #212121; }
        textarea { resize: none; }
    </style>
</head>

<body class="text-gray-200 h-screen flex overflow-hidden font-sans antialiased selection:bg-blue-500 selection:text-white relative">

    <!-- Sidebar -->
    <div class="hidden md:flex w-[260px] bg-[#171717] flex-col h-full border-r border-[#333]">
        <div class="p-4">
            <button onclick="startNewChat()" class="flex items-center gap-2 w-full p-3 rounded-lg hover:bg-[#2f2f2f] transition-colors border border-[#424242] text-sm text-gray-300 group">
                <i data-lucide="plus" class="w-4 h-4 text-gray-400 group-hover:text-white"></i>
                New Chat
            </button>
        </div>
        <div class="flex-1 overflow-y-auto p-4 pt-0">
            <p class="text-xs font-semibold text-gray-500 mb-3 px-2">Conversations</p>
            <div id="chat-list" class="flex flex-col space-y-1">
                <!-- Chat history loads here dynamically -->
            </div>
        </div>
        <div class="p-4 border-t border-[#333]">
            <button onclick="toggleSettings()" class="flex items-center gap-3 w-full p-2 rounded-lg hover:bg-[#2f2f2f] transition-colors text-sm text-gray-300">
                <div class="w-7 h-7 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold">U</div>
                User Settings
            </button>
        </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 flex flex-col h-full relative">
        
        <!-- Header (Mobile) -->
        <header class="flex md:hidden items-center justify-between p-4 border-b border-[#333] bg-[#212121] z-10">
            <button class="text-gray-400 hover:text-white"><i data-lucide="menu" class="w-6 h-6"></i></button>
            <h1 class="text-md font-medium text-gray-200">AI Assistant</h1>
            <button onclick="startNewChat()" class="text-gray-400 hover:text-white"><i data-lucide="plus" class="w-6 h-6"></i></button>
        </header>

        <!-- Chat Area -->
        <div id="chat-container" class="flex-1 overflow-y-auto w-full pb-36 pt-8 scroll-smooth">
            <div id="chat" class="max-w-3xl mx-auto flex flex-col space-y-6 px-4 md:px-0">
                <!-- Messages load here -->
            </div>
        </div>

        <!-- Input Area -->
        <div class="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#212121] via-[#212121] to-transparent pt-10 pb-6 px-4">
            <div class="max-w-3xl mx-auto relative">
                <div class="bg-[#2f2f2f] rounded-2xl border border-[#424242] focus-within:border-gray-500 focus-within:ring-1 focus-within:ring-gray-500 transition-all flex items-end p-2 shadow-lg">
                    
                    <button class="p-3 text-gray-400 hover:text-white rounded-xl transition-colors shrink-0">
                        <i data-lucide="paperclip" class="w-5 h-5"></i>
                    </button>
                    
                    <textarea 
                        id="input" 
                        rows="1"
                        placeholder="Message AI Assistant..." 
                        class="w-full max-h-[200px] bg-transparent text-gray-100 placeholder-gray-500 px-2 py-3 border-none outline-none focus:ring-0 text-md"
                    ></textarea>
                    
                    <button id="send-btn" onclick="send()" class="p-2 mb-1 mr-1 bg-white text-black hover:bg-gray-200 rounded-xl transition-colors shrink-0 disabled:opacity-50 disabled:cursor-not-allowed">
                        <i data-lucide="arrow-up" class="w-5 h-5"></i>
                    </button>

                </div>
                <div class="text-center mt-3 text-xs text-gray-500">
                    Advanced AI Assistant System. Model: <span class="font-semibold text-gray-400 border-b border-gray-600 border-dashed cursor-help" title="Configured in backend">phi3:mini</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div id="settings-modal" class="hidden fixed inset-0 bg-black/70 z-50 flex items-center justify-center backdrop-blur-sm transition-opacity">
        <div class="bg-[#171717] border border-[#333] rounded-2xl w-full max-w-lg p-6 shadow-2xl">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-xl font-semibold text-white flex items-center gap-2">
                    <i data-lucide="settings" class="w-5 h-5"></i> System Settings
                </h2>
                <button onclick="toggleSettings()" class="text-gray-400 hover:text-white"><i data-lucide="x" class="w-6 h-6"></i></button>
            </div>
            
            <div class="space-y-5">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">System Prompt (Persona)</label>
                    <textarea id="sys-prompt" rows="4" class="w-full bg-[#212121] text-gray-200 border border-[#424242] rounded-lg p-3 focus:border-blue-500 focus:outline-none transition-colors" placeholder="e.g. You are a helpful expert coding assistant who always replies in short concise sentences."></textarea>
                    <p class="text-xs text-gray-500 mt-2">Instructions placed here will be sent secretly to the AI before every conversation to guide its behavior.</p>
                </div>
                
                <button onclick="saveSettings()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors flex justify-center items-center gap-2">
                    <i data-lucide="save" class="w-4 h-4"></i> Save Settings
                </button>
            </div>
        </div>
    </div>

    <script>
        // Initialize Icons
        lucide.createIcons();

        // DOM Elements
        const chatEl = document.getElementById("chat");
        const chatContainer = document.getElementById("chat-container");
        const inputEl = document.getElementById("input");
        const sendBtn = document.getElementById("send-btn");
        const chatListEl = document.getElementById("chat-list");
        const settingsModal = document.getElementById("settings-modal");
        const sysPromptInput = document.getElementById("sys-prompt");

        // Application State
        let chats = JSON.parse(localStorage.getItem('ai_advanced_chats') || '[]');
        let currentChatId = null;
        let messageHistory = [];
        let systemPrompt = localStorage.getItem('ai_system_prompt') || '';

        // Auto-resize textarea
        inputEl.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight <= 200 ? this.scrollHeight : 200) + 'px';
            if (this.value.trim() === '') this.style.height = 'auto';
        });

        // Handle Enter key (Shift+Enter for new line)
        inputEl.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        });

        // ==========================================
        // Chat History & Persistence Management
        // ==========================================

        function initApp() {
            sysPromptInput.value = systemPrompt;
            if (chats.length === 0) {
                startNewChat();
            } else {
                renderSidebar();
                loadChat(chats[0].id); // Load most recent chat
            }
        }

        function startNewChat() {
            currentChatId = Date.now().toString();
            messageHistory = [];
            chats.unshift({ id: currentChatId, title: 'New Conversation', messages: [] });
            saveData();
            loadChat(currentChatId);
        }

        function loadChat(id) {
            currentChatId = id;
            const chat = chats.find(c => c.id === id);
            messageHistory = chat ? [...chat.messages] : [];
            chatEl.innerHTML = ""; 
            
            if (messageHistory.length === 0) {
                appendMessage("ai", "Hello! I'm fully operational and ready to assist. You can ask me questions, request code, or adjust my behavior in settings. How can I help you today?");
            } else {
                messageHistory.forEach(msg => appendMessage(msg.role, msg.content));
            }
            renderSidebar();
        }

        function deleteChat(id, event) {
            event.stopPropagation(); // prevent triggering loadChat
            chats = chats.filter(c => c.id !== id);
            if (chats.length === 0) {
                startNewChat();
            } else if (currentChatId === id) {
                loadChat(chats[0].id);
            } else {
                saveData();
            }
        }

        function saveData() {
            if(currentChatId) {
                const chatIndex = chats.findIndex(c => c.id === currentChatId);
                if(chatIndex > -1) {
                    chats[chatIndex].messages = [...messageHistory];
                    // Auto-generate title if it's new
                    if (chats[chatIndex].title === 'New Conversation' && messageHistory.length > 0) {
                        const firstMsg = messageHistory.find(m => m.role === 'user');
                        if (firstMsg) {
                            chats[chatIndex].title = firstMsg.content.substring(0, 25) + (firstMsg.content.length > 25 ? '...' : '');
                        }
                    }
                }
            }
            localStorage.setItem('ai_advanced_chats', JSON.stringify(chats));
            renderSidebar();
        }

        function renderSidebar() {
            chatListEl.innerHTML = '';
            chats.forEach(chat => {
                const isActive = chat.id === currentChatId;
                const btn = document.createElement('div');
                btn.className = `group flex items-center justify-between w-full p-2.5 rounded-lg text-sm cursor-pointer transition-colors ${isActive ? 'bg-[#2f2f2f] text-white' : 'hover:bg-[#212121] text-gray-400'}`;
                
                btn.innerHTML = `
                    <div class="flex items-center gap-3 overflow-hidden flex-1" onclick="loadChat('${chat.id}')">
                        <i data-lucide="message-square" class="w-4 h-4 shrink-0"></i>
                        <span class="truncate">${chat.title}</span>
                    </div>
                    <button class="shrink-0 p-1 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-opacity rounded" onclick="deleteChat('${chat.id}', event)">
                        <i data-lucide="trash-2" class="w-4 h-4"></i>
                    </button>
                `;
                chatListEl.appendChild(btn);
            });
            lucide.createIcons({ root: chatListEl });
        }

        // ==========================================
        // UI Interaction & Streaming Rendering
        // ==========================================

        function toggleSettings() {
            settingsModal.classList.toggle('hidden');
        }

        function saveSettings() {
            systemPrompt = sysPromptInput.value.trim();
            localStorage.setItem('ai_system_prompt', systemPrompt);
            toggleSettings();
        }

        function formatMarkdown(content) {
            const rawMarkup = marked.parse(content, { breaks: true, gfm: true });
            return DOMPurify.sanitize(rawMarkup, { ADD_CLASSES: {'code': 'hljs', 'pre': 'hljs'} });
        }

        function finalizeCodeBlocks(container) {
            container.querySelectorAll('pre').forEach((preBlock) => {
                if (preBlock.parentNode.classList.contains('group')) return; // Already wrapped
                
                // Highlight inner code
                const codeBlock = preBlock.querySelector('code');
                if(codeBlock) hljs.highlightElement(codeBlock);

                // Add wrapper and copy button
                const wrapper = document.createElement('div');
                wrapper.className = 'relative group mt-3 mb-3 rounded-lg overflow-hidden border border-[#333]';
                
                preBlock.parentNode.insertBefore(wrapper, preBlock);
                wrapper.appendChild(preBlock);
                preBlock.style.margin = '0'; // Remove default prose margin inside wrapper

                const copyBtn = document.createElement('button');
                copyBtn.className = 'absolute top-2 right-2 px-2 py-1.5 rounded bg-[#2f2f2f] text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5 text-xs font-medium border border-[#424242] shadow-sm';
                copyBtn.innerHTML = '<i data-lucide="copy" class="w-3 h-3"></i> Copy';
                copyBtn.onclick = () => {
                    navigator.clipboard.writeText(codeBlock ? codeBlock.innerText : preBlock.innerText);
                    copyBtn.innerHTML = '<i data-lucide="check" class="w-3 h-3 text-green-500"></i> Copied';
                    lucide.createIcons({ root: copyBtn });
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i data-lucide="copy" class="w-3 h-3"></i> Copy';
                        lucide.createIcons({ root: copyBtn });
                    }, 2000);
                };
                wrapper.appendChild(copyBtn);
            });
            lucide.createIcons({ root: container });
        }

        function appendMessage(role, content, id = null) {
            const div = document.createElement("div");
            div.className = "flex gap-4 w-full message-block";
            if (id) div.id = id;

            const isAI = role === 'ai';
            
            const avatarHtml = isAI 
                ? `<div class="w-8 h-8 rounded-full bg-white flex items-center justify-center shrink-0 shadow-sm"><i data-lucide="sparkles" class="w-5 h-5 text-black"></i></div>`
                : `<div class="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center shrink-0 text-white text-sm font-bold shadow-sm">U</div>`;

            let contentHtml = content;
            if (isAI && content !== '...') {
                contentHtml = formatMarkdown(content);
            } else if (!isAI) {
                contentHtml = content.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
            }

            if (content === '...') {
                contentHtml = `
                    <div class="flex items-center space-x-1 h-6">
                        <div class="w-1.5 h-1.5 bg-gray-400 rounded-full typing-dot"></div>
                        <div class="w-1.5 h-1.5 bg-gray-400 rounded-full typing-dot"></div>
                        <div class="w-1.5 h-1.5 bg-gray-400 rounded-full typing-dot"></div>
                    </div>
                `;
            }

            div.innerHTML = `
                ${avatarHtml}
                <div class="flex-1 min-w-0 pt-1">
                    <div class="prose prose-invert max-w-none text-[15px] text-gray-200 leading-relaxed content-container">
                        ${contentHtml}
                    </div>
                </div>
            `;

            chatEl.appendChild(div);
            lucide.createIcons({ root: div });
            
            // Post-processing for loaded history
            if (isAI && content !== '...') {
                finalizeCodeBlocks(div);
            }

            scrollToBottom();
            return div;
        }

        function scrollToBottom() {
            chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
        }

        async function send() {
            const text = inputEl.value.trim();
            if (!text) return;

            // Add user message to history and save
            messageHistory.push({ role: "user", content: text });
            saveData();

            // Reset input UI
            inputEl.value = "";
            inputEl.style.height = 'auto';
            inputEl.disabled = true;
            sendBtn.disabled = true;

            appendMessage("user", text);
            const loadingId = "loading-" + Date.now();
            const messageDiv = appendMessage("ai", "...", loadingId);
            const contentContainer = messageDiv.querySelector('.content-container');

            try {
                // Prepare payload with system prompt if it exists
                let payloadMessages = [...messageHistory];
                if (systemPrompt) {
                    payloadMessages.unshift({ role: "system", content: systemPrompt });
                }

                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ messages: payloadMessages })
                });

                if (!response.ok) throw new Error("Network error");

                // Stream processing
                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let accumulatedResponse = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    accumulatedResponse += chunk;

                    // Update UI live as chunks stream in
                    contentContainer.innerHTML = formatMarkdown(accumulatedResponse);
                    scrollToBottom();
                }

                // Finalize specific elements (e.g., Code Highlight & Copy Buttons)
                finalizeCodeBlocks(messageDiv);
                
                // Add AI reply to history and save
                messageHistory.push({ role: "assistant", content: accumulatedResponse });
                saveData();

            } catch (error) {
                contentContainer.innerHTML = formatMarkdown("**Error:** Failed to connect to the AI API. Check your network or server status.");
                messageHistory.pop(); // Revert user message on fail
                saveData();
            } finally {
                inputEl.disabled = false;
                sendBtn.disabled = false;
                inputEl.focus();
            }
        }

        // Initialize application on load
        window.onload = initApp;

    </script>
</body>
</html>
"""

def ask_ai_stream(messages):
    try:
        # Enable stream=True to continuously receive data chunks from the model
        response = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True # Command the API to stream back
            },
            stream=True,       # Let python requests handle the stream chunks
            timeout=60
        )
        
        response.raise_for_status()

        # Iterate over new lines representing streaming chunks
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    data = json.loads(decoded_line)
                    # Extract content chunks natively from the JSON lines
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                    elif "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.Timeout:
        yield "\n\n**Error:** Request timed out. The model took too long to respond."
    except Exception as e:
        yield f"\n\n**Error:** Connection error: {str(e)}"

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    messages = request.json.get("messages", [])
    # Return a streamed plain-text response that our Javascript client will assemble chunk-by-chunk
    return Response(stream_with_context(ask_ai_stream(messages)), mimetype='text/plain')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
