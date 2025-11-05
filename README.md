# PrivateLLM
The ultimate self-hosted, privacy-first web interface for Ollama. Features Auto Mode, ChatGPT import, code artifacts, and complete data sovereignty.
Here is a detailed `README.md` for your advanced Ollama chat application, based on the provided `server.py` and `index.html` files.

-----

# PrivateLLM Advanced WebUI

This is a powerful, self-hosted web interface for interacting with local language models via **Ollama**. It's built with Flask and vanilla JavaScript, offering advanced features designed for developers and power users, including smart model switching, autonomous agents, file uploads, and rich text rendering.

This application is **100% local**. Your models, settings, and chat history never leave your machine, ensuring complete data sovereignty and privacy.

## ✨ Key Features

This application goes beyond a simple chat interface, offering a suite of intelligent tools:

### Intelligent & Autonomous Features

  * **🤖 Smart Model Switching:** Automatically selects the best-suited *available* model for your prompt. For example, it heuristics to use models like `codellama` for coding tasks, larger models for complex analysis, or fast models for simple questions. This feature can be toggled in the settings.
  * **🚀 Auto Mode:** A true autonomous agent mode. When activated, the AI will evaluate its own work, identify the next logical step, and continue executing in a loop until the task is complete or it reaches the max iteration limit.
  * **✍️ AI-Powered Title Generation:** Instead of just using the first few words of your prompt, a lightweight LLM call generates a concise, descriptive title for every new chat (e.g., "Python Data Analysis Script").

### Rich Chat & Data Interface

  * **📎 File Uploads:** Attach text-based files (`.txt`, `.py`, `.js`, `.md`, `.pdf`, `.doc`, etc.) to your messages. The application reads the text content and includes it in the prompt, perfect for code reviews or document analysis.
  * **🧮 Rich Markdown & LaTeX:** Responses are beautifully rendered with support for full markdown, including tables, lists, and blockquotes.
  * **Syntax Highlighting:** All code blocks are automatically highlighted with `highlight.js`, complete with a "Copy code" button.
  * **LaTeX Math Rendering:** Mathematical formulas and equations are displayed clearly using KaTeX.
  * **Chat Export:** Export any chat conversation to a clean Markdown file.

### Advanced Session & Model Management

  * **Message Editing:** Edit your previous prompts and resend them, forking the conversation from that point.
  * **Multi-Mode Chat:** Instantly switch between specialized system prompts for different tasks:
      * `Chat`: Standard, helpful conversation.
      * `Code`: For expert-level programming assistance.
      * `Auto`: The autonomous agent mode.
      * `Project`: For architecting multi-file projects.
  * **In-App Model Management:** From the settings modal, you can:
      * View all locally installed Ollama models and their sizes.
      * Delete any model from your machine.
      * Download new models from the Ollama library using a search bar or quick-suggestion buttons.

## 🚀 Getting Started

### Prerequisites

1.  **Ollama:** You must have the [Ollama server](https://ollama.com/) installed and running on your system.
2.  **Python 3:** The backend is powered by Python.
3.  **Flask:** The only Python dependency.

### Installation & Running

1.  **Install Flask:**

    ```bash
    pip install flask
    ```

2.  **Start the Ollama Server:** Open a terminal and ensure the Ollama service is running.

    ```bash
    ollama serve
    ```

3.  **Install Models:** Make sure you have at least one model installed. The application defaults to `llama3.2`, but will work with any model.

    ```bash
    ollama pull llama3.2
    ollama pull codellama  # Recommended for Smart Model Switching
    ```

4.  **Run the Application Server:** Save the provided files as `server.py` and `index.html` in the same directory. Then, run the Python script:

    ```bash
    python server.py
    ```

5.  **Open in Browser:** Navigate to `http://localhost:5000` in your web browser to start chatting.

## 🔧 Configuration

### In-App Settings

You can configure the application's behavior by clicking the **"Settings" (⚙)** icon.

  * **Model Configuration:**

      * **Default Model:** The fallback model to use if Smart Switching is off.
      * **Temperature:** Controls the creativity of the AI (0.0 = deterministic, 2.0 = very creative).
      * **Max Tokens:** The maximum length of a generated response.
      * **Context Length:** How much conversation history the model "remembers".

  * **Smart Features:**

      * **Smart Model Switching:** Toggle the automatic model selection feature.
      * **Auto Mode Iterations:** Enable or disable the autonomous loop for "Auto Mode".
      * **Max Auto Iterations:** Set a limit (e.g., 3-10) to prevent infinite loops.

  * **System Prompt:**

      * Add a global system prompt to customize the AI's personality or rules for all modes.

### Data Storage

This application is designed for privacy. All your data is stored locally in your home directory:

  * **Storage Path:** `~/.ollama_chat/`
  * **Chats:** `chats.json` contains your complete conversation history.
  * **Settings:** `settings.json` stores your application preferences.

## 💻 Tech Stack

  * **Backend:** Flask (Python)
  * **Frontend:** Vanilla JavaScript (ES6+), HTML5, CSS3
  * **Markdown:** `marked.js`
  * **Syntax Highlighting:** `highlight.js`
  * **LaTeX Rendering:** `katex`
