# 🌌 OmniNexus: The Next-Gen AI Swarm Orchestrator



[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

**OmniNexus** is a state-of-the-art multi-agent AI orchestration platform designed for high-performance visual intelligence and autonomous task execution. Built with a sleek glassmorphic interface, it leverages a sophisticated "Swarm Router" to coordinate specialized AI nodes across research, coding, system analysis, and more.

---

## 🚀 Key Capabilities

| Feature | Description |
| :--- | :--- |
| **🧠 Swarm Intelligence** | A multi-node routing engine that dynamically assigns tasks to specialized nodes (Explorer, Coder, Analyzer). |
| **🌐 Deep Web Research** | Real-time browsing using Playwright and Tavily API for deep, fact-checked research reports. |
| **📁 Local RAG & GraphRAG** | Advanced document retrieval powered by ChromaDB and a persistent SQLite Knowledge Graph. |
| **🛰️ 3D Brain Map** | Interactive 3D visualization of the AI's long-term memory and entity relationships. |
| **⚡ Technical Execution** | Native support for Python scripting, system shell commands, and local file management. |
| **🎨 Premium UI/UX** | A stunning glassmorphic dashboard with ambient animations and real-time streaming. |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **AI Brain:** Ollama (Gemma, Qwen, Llava), ChromaDB (Vector Store)
- **Knowledge Graph:** SQLite + ForceGraph3D
- **Interface:** HTML5, Tailwind CSS, JavaScript (ES6+)
- **Automation:** Playwright (Live Web Agent)
- **Search:** Tavily API / DuckDuckGo Lite

---

## 📦 Getting Started

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Ollama](https://ollama.ai/) (for local model hosting)
- [Node.js](https://nodejs.org/) (optional, for advanced web agent features)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mdmusab123/AI-Swarm-Orchestrator.git
   cd AI-Swarm-Orchestrator
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m playwright install chromium
   ```

3. **Set Up Models:**
   Ensure Ollama is running and pull the necessary models:
   ```bash
   ollama pull gemma
   ollama pull qwen2:0.5b
   ollama pull llava
   ```

4. **Launch OmniNexus:**
   ```bash
   python main.py
   ```
   Access the dashboard at `http://127.0.0.1:5000`.

---

## 🧩 Advanced Features

### Interactive 3D Brain Map
OmniNexus features a unique **Memory Constellation**—a 3D interactive graph that visualizes how the AI connects concepts, entities, and long-term memories in real-time.

### Autonomous Swarm Routing
The project uses a high-speed nano-model (Qwen 0.5b) as a router, ensuring near-instantaneous task classification without the overhead of larger models.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built with ❤️ for the future of Agentic AI.
</p>
