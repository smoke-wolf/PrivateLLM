from flask import Flask, request, jsonify, send_file, send_from_directory
import json
import os
from datetime import datetime
import subprocess
from pathlib import Path
import tempfile
import re
import threading

app = Flask(__name__, static_folder='.')

# Data storage
DATA_DIR = Path.home() / ".ollama_chat"
DATA_DIR.mkdir(exist_ok=True)
CHATS_FILE = DATA_DIR / "chats.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

def load_chats():
    if CHATS_FILE.exists():
        try:
            with open(CHATS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_chats(chats):
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=2)

def load_settings():
    defaults = {
        "model": "llama3.2",
        "temperature": 0.7,
        "system_prompt": "",
        "max_tokens": 2048,
        "context_length": 4096,
        "smart_model_switch": True,
        "auto_iterations": True,
        "max_iterations": 3
    }
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except:
            pass
    return defaults

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# PROPRIETARY FEATURE 1: Smart Model Switching
def select_best_model(message, available_models):
    """
    Intelligently selects the best model based on message content.
    Uses heuristics to match task complexity with model capabilities.
    """
    message_lower = message.lower()
    
    # Check for code-related tasks
    code_keywords = ['code', 'function', 'class', 'debug', 'program', 'script', 'implement']
    if any(keyword in message_lower for keyword in code_keywords):
        # Prefer code-specialized models
        for model in available_models:
            if 'codellama' in model['name'] or 'deepseek' in model['name'] or 'coder' in model['name']:
                return model['name']
    
    # Check for complex reasoning tasks
    complex_keywords = ['analyze', 'compare', 'explain', 'elaborate', 'complex', 'detailed']
    if any(keyword in message_lower for keyword in complex_keywords) or len(message) > 500:
        # Prefer larger models
        for model in available_models:
            if ':70b' in model['name'] or ':34b' in model['name'] or '3.1' in model['name']:
                return model['name']
    
    # Check for math/reasoning
    math_keywords = ['calculate', 'math', 'equation', 'solve', 'proof']
    if any(keyword in message_lower for keyword in math_keywords):
        for model in available_models:
            if 'qwen' in model['name'] or 'mistral' in model['name']:
                return model['name']
    
    # Default to fastest model for simple queries
    if len(message) < 100:
        for model in available_models:
            if '3.2' in model['name'] or ':3b' in model['name']:
                return model['name']
    
    # Fallback to first available model
    return available_models[0]['name'] if available_models else 'llama3.2'

# PROPRIETARY FEATURE 2: Context-Aware Title Generation
def generate_title_from_message(message, model):
    """
    Uses AI to generate concise, meaningful chat titles.
    Analyzes the message intent and creates a descriptive title.
    """
    try:
        prompt = f"""Generate a very short title (maximum 5 words) for a conversation starting with this message. 
The title should capture the main topic or question. Respond with ONLY the title, nothing else.

Message: {message[:300]}

Title:"""
        
        process = subprocess.Popen(
            ['ollama', 'run', model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=prompt, timeout=15)
        title = stdout.strip()
        
        # Clean the title
        title = title.replace('"', '').replace("'", "").strip()
        title = title.split('\n')[0][:60]
        
        # Remove common prefixes
        for prefix in ['Title:', 'title:', 'TITLE:']:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        if not title or len(title) < 3:
            title = message.strip().split('\n')[0][:50]
            if len(message) > 50:
                title += '...'
        
        return title
    except Exception as e:
        print(f"Title generation error: {e}")
        title = message.strip().split('\n')[0][:50]
        if len(message) > 50:
            title += '...'
        return title

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chats', methods=['GET'])
def get_chats():
    chats = load_chats()
    return jsonify(chats)

@app.route('/api/chats', methods=['POST'])
def create_chat():
    chats = load_chats()
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().timestamp(),
        "updated": datetime.now().timestamp(),
        "model": load_settings()["model"]
    }
    save_chats(chats)
    return jsonify({"chat_id": chat_id, "chat": chats[chat_id]})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        return jsonify(chats[chat_id])
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chats/<chat_id>/export', methods=['GET'])
def export_chat(chat_id):
    chats = load_chats()
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    chat = chats[chat_id]
    
    # Create markdown export
    md_content = f"# {chat['title']}\n\n"
    md_content += f"**Created:** {datetime.fromtimestamp(chat['created']).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"
    
    for msg in chat['messages']:
        role = msg['role'].upper()
        content = msg['content']
        model = msg.get('model', '')
        if model:
            md_content += f"## {role} ({model})\n\n{content}\n\n---\n\n"
        else:
            md_content += f"## {role}\n\n{content}\n\n---\n\n"
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
    temp_file.write(md_content)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"{chat['title'][:30]}.md"
    )

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    chats = load_chats()
    if chat_id not in chats:
        return jsonify({"error": "Chat not found"}), 404
    
    data = request.json
    message = data.get('message')
    mode = data.get('mode', 'chat')
    smart_model = data.get('smart_model', False)
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    settings = load_settings()
    
    # Smart model selection
    selected_model = settings["model"]
    if smart_model:
        try:
            models_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            lines = models_result.stdout.strip().split('\n')[1:]
            available_models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        available_models.append({"name": parts[0], "size": parts[1] if len(parts) > 1 else "Unknown"})
            
            if available_models:
                selected_model = select_best_model(message, available_models)
                print(f"Smart model selection: {selected_model}")
        except Exception as e:
            print(f"Smart model selection failed: {e}")
    
    # Add user message
    chats[chat_id]["messages"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().timestamp()
    })
    
    # Generate chat title from first message
    if len(chats[chat_id]["messages"]) == 1:
        title = generate_title_from_message(message, selected_model)
        chats[chat_id]["title"] = title
    
    chats[chat_id]["updated"] = datetime.now().timestamp()
    save_chats(chats)
    
    # Generate response
    try:
        # Prepare system prompt based on mode
        system_prompt = settings.get("system_prompt", "")
        auto_thinking = None
        
        if mode == "code":
            system_prompt = """You are an expert programmer. Provide clear, working code with explanations. 
When writing code:
1. Use proper syntax highlighting by specifying the language
2. Include comments explaining complex parts
3. Provide complete, runnable examples
4. Format code in markdown code blocks with language tags"""
        
        elif mode == "auto":
            # Auto mode: Autonomous iteration with self-evaluation
            system_prompt = """You are an autonomous AI assistant working on a task. For each response:
1. Evaluate what has been accomplished so far
2. Identify the next logical step
3. Execute that step thoroughly
4. State clearly if the task is COMPLETE or if more work is needed

Think step-by-step and work systematically. Break complex tasks into manageable pieces.
After each step, briefly state what you did and what comes next."""
            
            # Check context for auto mode thinking
            recent_messages = chats[chat_id]["messages"][-3:]
            if len(recent_messages) > 1:
                auto_thinking = "Iteration in progress..."
        
        elif mode == "project":
            system_prompt = """You are a project architect and developer. When given a project:
1. Break it down into components and files
2. Create a clear structure
3. Implement with complete, working code
4. Include tests and documentation
5. Format files as:
```language
# File: path/to/file.ext
[code here]
```"""
        
        # Build conversation context
        context_messages = chats[chat_id]["messages"][-10:]
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        for msg in context_messages:
            role_prefix = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role_prefix}: {msg['content']}\n\n"
        
        full_prompt += "Assistant:"
        
        # Call ollama
        process = subprocess.Popen(
            ['ollama', 'run', selected_model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=full_prompt, timeout=120)
        response = stdout.strip()
        
        if not response:
            response = "Error: No response from model. Make sure Ollama is running and the model is installed."
        
        # Add assistant response
        chats[chat_id]["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().timestamp(),
            "model": selected_model,
            "auto_thinking": auto_thinking
        })
        chats[chat_id]["updated"] = datetime.now().timestamp()
        save_chats(chats)
        
        return jsonify({
            "response": response,
            "chat": chats[chat_id],
            "model": selected_model,
            "auto_thinking": auto_thinking
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Request timeout - try a simpler prompt"}), 408
    except FileNotFoundError:
        return jsonify({"error": "Ollama not found. Make sure it's installed and running."}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(load_settings())

@app.route('/api/settings', methods=['POST'])
def update_settings():
    settings = request.json
    save_settings(settings)
    return jsonify({"success": True})

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split('\n')[1:]
        models = []
        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append({
                        "name": model_name,
                        "size": parts[1] if len(parts) > 1 else "Unknown"
                    })
        
        if not models:
            models = [{"name": "llama3.2", "size": "2GB"}]
        return jsonify({"models": models})
    except Exception as e:
        print(f"Error loading models: {e}")
        return jsonify({"models": [{"name": "llama3.2", "size": "2GB"}]})

@app.route('/api/models/download', methods=['POST'])
def download_model():
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"error": "No model specified"}), 400
    
    try:
        # Start download in background
        def download():
            try:
                subprocess.run(['ollama', 'pull', model_name], check=True, timeout=3600)
            except Exception as e:
                print(f"Model download error: {e}")
        
        thread = threading.Thread(target=download, daemon=True)
        thread.start()
        
        return jsonify({"success": True, "message": f"Downloading {model_name}..."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({"error": "No model specified"}), 400
    
    try:
        subprocess.run(['ollama', 'rm', model_name], check=True, timeout=30)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Ollama Chat Server Starting...")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Server: http://localhost:5000")
    print("\nFeatures:")
    print("  • Smart Model Switching - Auto-selects best model")
    print("  • Auto Mode Iterations - Autonomous task completion")
    print("  • File Upload Support - Analyze any text file")
    print("  • LaTeX Rendering - Beautiful math equations")
    print("=" * 60)
    print("\nMake sure Ollama is running:")
    print("   ollama serve")
    print("\nInstall models:")
    print("   ollama pull llama3.2")
    print("   ollama pull codellama")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, port=5000, threaded=True)
