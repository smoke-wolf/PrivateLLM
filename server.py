from flask import Flask, request, jsonify, send_file, send_from_directory
import json
import os
from datetime import datetime
import subprocess
from pathlib import Path
import tempfile
import re
import threading
import time
from typing import Dict, List, Optional
import uuid

app = Flask(__name__, static_folder='.')

# Data storage
DATA_DIR = Path.home() / ".ollama_chat"
DATA_DIR.mkdir(exist_ok=True)
CHATS_FILE = DATA_DIR / "chats.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
STATS_FILE = DATA_DIR / "generation_stats.json"
CHATGPT_INDEX_FILE = DATA_DIR / "chatgpt_index.json"

# ChatGPT export path (configurable)
CHATGPT_EXPORT_PATH = Path.home() / "Downloads"

# In-memory cache for ChatGPT conversations
_chatgpt_cache = {}
_chatgpt_export_files = []

# ============================================================================
# GENERATION STATS TRACKING
# ============================================================================

def load_stats():
    """Load generation statistics."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"generations": []}
    return {"generations": []}

def save_stats(stats):
    """Save generation statistics."""
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def record_generation(chat_id, model, prompt_length, response_length, duration, tokens_per_second=None):
    """Record a generation event with timing data."""
    stats = load_stats()
    
    generation_record = {
        "timestamp": datetime.now().timestamp(),
        "chat_id": chat_id,
        "model": model,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "duration_seconds": round(duration, 2),
        "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else None,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    stats["generations"].append(generation_record)
    
    # Keep only last 1000 records to prevent file bloat
    if len(stats["generations"]) > 1000:
        stats["generations"] = stats["generations"][-1000:]
    
    save_stats(stats)
    return generation_record

def get_generation_stats_summary():
    """Get summary statistics for generations."""
    stats = load_stats()
    generations = stats.get("generations", [])
    
    if not generations:
        return {
            "total_generations": 0,
            "average_duration": 0,
            "average_tokens_per_second": 0,
            "by_model": {}
        }
    
    by_model = {}
    for gen in generations:
        model = gen.get("model", "unknown")
        if model not in by_model:
            by_model[model] = {
                "count": 0,
                "total_duration": 0,
                "total_tokens": 0,
                "durations": []
            }
        
        by_model[model]["count"] += 1
        by_model[model]["total_duration"] += gen.get("duration_seconds", 0)
        by_model[model]["durations"].append(gen.get("duration_seconds", 0))
        if gen.get("tokens_per_second"):
            by_model[model]["total_tokens"] += gen.get("tokens_per_second", 0)
    
    # Calculate averages
    for model in by_model:
        count = by_model[model]["count"]
        by_model[model]["average_duration"] = round(by_model[model]["total_duration"] / count, 2)
        by_model[model]["average_tokens_per_second"] = round(by_model[model]["total_tokens"] / count, 2) if by_model[model]["total_tokens"] > 0 else 0
        by_model[model]["median_duration"] = round(sorted(by_model[model]["durations"])[count // 2], 2)
        del by_model[model]["durations"]
        del by_model[model]["total_duration"]
        del by_model[model]["total_tokens"]
    
    total_duration = sum(g.get("duration_seconds", 0) for g in generations)
    
    return {
        "total_generations": len(generations),
        "average_duration": round(total_duration / len(generations), 2),
        "by_model": by_model,
        "recent_generations": generations[-10:]
    }

# ============================================================================
# CHATGPT IMPORT FUNCTIONS (WITH LAZY LOADING)
# ============================================================================

def parse_chatgpt_metadata(conv_data: Dict) -> Optional[Dict]:
    """
    Extract only metadata from a ChatGPT conversation (no messages).
    This is used for fast initial loading.
    """
    try:
        conv_id = conv_data.get('id') or conv_data.get('conversation_id')
        if not conv_id:
            return None
        
        # Extract only metadata
        title = conv_data.get('title', 'Untitled')
        create_time = conv_data.get('create_time')
        update_time = conv_data.get('update_time')
        
        # Count messages without parsing them
        mapping = conv_data.get('mapping', {})
        message_count = sum(1 for node in mapping.values() if node.get('message'))
        
        return {
            'id': f"chatgpt_{conv_id}",
            'title': title,
            'created': create_time or datetime.now().timestamp(),
            'updated': update_time or datetime.now().timestamp(),
            'source': 'chatgpt',
            'model': conv_data.get('default_model_slug', 'gpt-4'),
            'original_id': conv_id,
            'message_count': message_count,
            'loaded': False  # Flag to indicate full content not loaded
        }
    
    except Exception as e:
        print(f"Error parsing ChatGPT metadata: {e}")
        return None


def parse_chatgpt_conversation_full(conv_data: Dict) -> Optional[Dict]:
    """
    Parse a complete ChatGPT conversation including all messages.
    This is called on-demand when a user opens a chat.
    """
    try:
        conv_id = conv_data.get('id') or conv_data.get('conversation_id')
        if not conv_id:
            return None
        
        # Extract conversation metadata
        title = conv_data.get('title', 'Untitled')
        create_time = conv_data.get('create_time')
        update_time = conv_data.get('update_time')
        
        # Parse the message tree from mapping
        mapping = conv_data.get('mapping', {})
        if not mapping:
            return None
        
        # Find the current node (end of conversation)
        current_node = conv_data.get('current_node')
        
        # Reconstruct conversation path from current_node backwards
        message_chain = []
        if current_node and current_node in mapping:
            node_id = current_node
            visited = set()
            
            # Traverse backwards to root
            while node_id and node_id not in visited:
                visited.add(node_id)
                node = mapping.get(node_id)
                if not node:
                    break
                
                message_chain.insert(0, node)
                node_id = node.get('parent')
        
        # Extract messages from chain
        messages = []
        for node in message_chain:
            msg = node.get('message')
            if not msg:
                continue
            
            author = msg.get('author', {})
            role = author.get('role', 'user')
            content = msg.get('content', {})
            
            # Extract text content
            parts = content.get('parts', [])
            if not parts:
                continue
            
            # Join all text parts
            text_content = '\n'.join(str(part) for part in parts if part)
            if not text_content.strip():
                continue
            
            # Map ChatGPT roles to Ollama format
            if role == 'assistant':
                role = 'assistant'
            elif role == 'system':
                role = 'system'
            else:
                role = 'user'
            
            messages.append({
                'role': role,
                'content': text_content,
                'timestamp': msg.get('create_time'),
                'model': conv_data.get('default_model_slug', 'gpt-4')
            })
        
        if not messages:
            return None
        
        return {
            'id': f"chatgpt_{conv_id}",
            'title': title,
            'created': create_time or datetime.now().timestamp(),
            'updated': update_time or datetime.now().timestamp(),
            'messages': messages,
            'source': 'chatgpt',
            'model': conv_data.get('default_model_slug', 'gpt-4'),
            'original_id': conv_id,
            'loaded': True
        }
    
    except Exception as e:
        print(f"Error parsing ChatGPT conversation: {e}")
        return None


def build_chatgpt_index(filepath: Path) -> Dict[str, Dict]:
    """
    Build an index of ChatGPT conversations with metadata only.
    Returns a dict mapping chat_id -> metadata.
    """
    if not filepath.exists():
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        index = {}
        
        # Handle both list and dict formats
        if isinstance(data, list):
            conversations = data
        elif isinstance(data, dict):
            conversations = []
            for key in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                conversations.append(data[key])
        else:
            return {}
        
        for conv in conversations:
            metadata = parse_chatgpt_metadata(conv)
            if metadata:
                index[metadata['id']] = metadata
        
        return index
    
    except Exception as e:
        print(f"Error building ChatGPT index: {e}")
        return {}


def load_chatgpt_conversation_by_id(chat_id: str) -> Optional[Dict]:
    """
    Load a specific ChatGPT conversation by ID from the export files.
    This is called on-demand when a user opens a chat.
    """
    # Check cache first
    if chat_id in _chatgpt_cache:
        return _chatgpt_cache[chat_id]
    
    # Extract original ID
    if not chat_id.startswith('chatgpt_'):
        return None
    
    original_id = chat_id[8:]  # Remove 'chatgpt_' prefix
    
    # Search through export files
    for export_file in _chatgpt_export_files:
        try:
            with open(export_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            conversations = []
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict):
                for key in data.keys():
                    conversations.append(data[key])
            
            # Find the conversation
            for conv in conversations:
                conv_id = conv.get('id') or conv.get('conversation_id')
                if conv_id == original_id:
                    # Parse full conversation
                    full_chat = parse_chatgpt_conversation_full(conv)
                    if full_chat:
                        # Cache it
                        _chatgpt_cache[chat_id] = full_chat
                        return full_chat
        
        except Exception as e:
            print(f"Error loading ChatGPT conversation from {export_file}: {e}")
            continue
    
    return None


def find_chatgpt_exports(base_path: Path = CHATGPT_EXPORT_PATH) -> List[Path]:
    """Find all conversations.json files in Downloads."""
    found = []
    if base_path.exists():
        # Look for direct conversations.json
        direct_file = base_path / "conversations.json"
        if direct_file.exists():
            found.append(direct_file)
        
        # Look in subdirectories (ChatGPT exports are in folders)
        for item in base_path.iterdir():
            if item.is_dir():
                conv_file = item / "conversations.json"
                if conv_file.exists():
                    found.append(conv_file)
    
    return found


def save_chatgpt_index(index: Dict):
    """Save ChatGPT conversation index to disk."""
    with open(CHATGPT_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def load_chatgpt_index() -> Dict[str, Dict]:
    """Load ChatGPT conversation index from disk."""
    if CHATGPT_INDEX_FILE.exists():
        try:
            with open(CHATGPT_INDEX_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}


def refresh_chatgpt_index():
    """
    Refresh the ChatGPT index by scanning export files.
    This should be called on startup and when explicitly requested.
    """
    global _chatgpt_export_files
    
    export_files = find_chatgpt_exports()
    _chatgpt_export_files = export_files
    
    all_metadata = {}
    
    for export_file in export_files:
        print(f"Indexing ChatGPT export: {export_file}")
        metadata = build_chatgpt_index(export_file)
        all_metadata.update(metadata)
    
    # Save index to disk
    save_chatgpt_index(all_metadata)
    
    print(f"✓ Indexed {len(all_metadata)} ChatGPT conversations")
    return all_metadata


def convert_chatgpt_to_ollama(chat_id: str) -> str:
    """
    Convert a ChatGPT chat to an editable Ollama chat.
    Creates a new chat with a new ID and copies all messages.
    """
    # First, ensure the full conversation is loaded
    chatgpt_chat = load_chatgpt_conversation_by_id(chat_id)
    
    if not chatgpt_chat:
        return None
    
    if not chat_id.startswith('chatgpt_'):
        return chat_id  # Already an Ollama chat
    
    # Create new Ollama chat ID
    new_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Copy chat data
    new_chat = {
        "title": chatgpt_chat['title'] + " (Imported)",
        "messages": chatgpt_chat['messages'].copy(),
        "created": datetime.now().timestamp(),
        "updated": datetime.now().timestamp(),
        "model": load_settings()["model"],
        "source": "ollama",
        "imported_from": chat_id
    }
    
    # Save the new chat
    ollama_chats = load_ollama_chats()
    ollama_chats[new_chat_id] = new_chat
    save_chats_direct(ollama_chats)
    
    return new_chat_id


def load_ollama_chats() -> Dict[str, Dict]:
    """Load existing Ollama chats."""
    if not CHATS_FILE.exists():
        return {}
    
    try:
        with open(CHATS_FILE, 'r', encoding='utf-8') as f:
            chats = json.load(f)
        return chats
    except json.JSONDecodeError:
        print("⚠ Error loading Ollama chats file (corrupted JSON)")
        return {}
    except Exception as e:
        print(f"⚠ Error loading Ollama chats: {e}")
        return {}


def load_chats() -> Dict[str, Dict]:
    """
    Load chat metadata from both Ollama format and ChatGPT index.
    ChatGPT chats only include metadata (titles, dates) - full content
    is loaded on-demand.
    """
    # Load Ollama chats (full data)
    ollama_chats = load_ollama_chats()
    
    # Load ChatGPT index (metadata only)
    chatgpt_index = load_chatgpt_index()
    
    # Merge (Ollama chats take precedence if there are conflicts)
    all_chats = {**chatgpt_index, **ollama_chats}
    
    return all_chats


# ============================================================================
# ORIGINAL FUNCTIONS (with updates)
# ============================================================================

def save_chats_direct(chats):
    """Save chats directly (for internal use)."""
    with open(CHATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def save_chats(chats):
    """
    Save chats to Ollama format.
    Filters out ChatGPT imports before saving.
    """
    # Filter out ChatGPT imports before saving
    ollama_chats = {k: v for k, v in chats.items() if not k.startswith('chatgpt_')}
    save_chats_direct(ollama_chats)

def load_settings():
    defaults = {
        "model": "llama3.2",
        "temperature": 0.7,
        "system_prompt": "",
        "max_tokens": 2048,
        "context_length": 4096,
        "smart_model_switch": True,
        "auto_iterations": True,
        "max_iterations": 3,
        "chatgpt_import_enabled": True,
        "track_generation_stats": True
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
    """Intelligently selects the best model based on message content."""
    message_lower = message.lower()
    
    # Check for code-related tasks
    code_keywords = ['code', 'function', 'class', 'debug', 'program', 'script', 'implement']
    if any(keyword in message_lower for keyword in code_keywords):
        for model in available_models:
            if 'codellama' in model['name'] or 'deepseek' in model['name'] or 'coder' in model['name']:
                return model['name']
    
    # Check for complex reasoning tasks
    complex_keywords = ['analyze', 'compare', 'explain', 'elaborate', 'complex', 'detailed']
    if any(keyword in message_lower for keyword in complex_keywords) or len(message) > 500:
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
    
    return available_models[0]['name'] if available_models else 'llama3.2'

# PROPRIETARY FEATURE 2: Context-Aware Title Generation
def generate_title_from_message(message, model):
    """Uses AI to generate concise, meaningful chat titles."""
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

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get all chat metadata (fast - no full message loading)."""
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
        "model": load_settings()["model"],
        "source": "ollama"
    }
    save_chats(chats)
    return jsonify({"chat_id": chat_id, "chat": chats[chat_id]})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a specific chat with full content (lazy loads ChatGPT chats)."""
    if chat_id.startswith('chatgpt_'):
        # Lazy load the full ChatGPT conversation
        chat = load_chatgpt_conversation_by_id(chat_id)
        if chat:
            return jsonify(chat)
        return jsonify({"error": "Chat not found"}), 404
    else:
        # Load from Ollama chats
        ollama_chats = load_ollama_chats()
        if chat_id in ollama_chats:
            return jsonify(ollama_chats[chat_id])
        return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chats/<chat_id>/convert', methods=['POST'])
def convert_chat(chat_id):
    """Convert a ChatGPT chat to an editable Ollama chat."""
    new_chat_id = convert_chatgpt_to_ollama(chat_id)
    
    if not new_chat_id:
        return jsonify({"error": "Chat not found"}), 404
    
    ollama_chats = load_ollama_chats()
    return jsonify({
        "success": True,
        "new_chat_id": new_chat_id,
        "chat": ollama_chats[new_chat_id]
    })

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    chats = load_ollama_chats()
    if chat_id in chats:
        del chats[chat_id]
        save_chats_direct(chats)
        return jsonify({"success": True})
    
    # Don't allow deleting ChatGPT imports (they're read-only)
    if chat_id.startswith('chatgpt_'):
        return jsonify({"error": "Cannot delete imported ChatGPT chats. Convert to Ollama chat first."}), 403
    
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chats/<chat_id>/export', methods=['GET'])
def export_chat(chat_id):
    # Load full chat (handles lazy loading)
    if chat_id.startswith('chatgpt_'):
        chat = load_chatgpt_conversation_by_id(chat_id)
    else:
        ollama_chats = load_ollama_chats()
        chat = ollama_chats.get(chat_id)
    
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    
    # Create markdown export
    md_content = f"# {chat['title']}\n\n"
    md_content += f"**Created:** {datetime.fromtimestamp(chat['created']).strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    source = chat.get('source', 'ollama')
    if source == 'chatgpt':
        md_content += f"**Source:** ChatGPT Import\n"
    
    md_content += "\n---\n\n"
    
    for msg in chat['messages']:
        role = msg['role'].upper()
        content = msg['content']
        model = msg.get('model', '')
        duration = msg.get('generation_time')
        
        if model:
            header = f"## {role} ({model})"
            if duration:
                header += f" - {duration}s"
            md_content += f"{header}\n\n{content}\n\n---\n\n"
        else:
            md_content += f"## {role}\n\n{content}\n\n---\n\n"
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
    temp_file.write(md_content)
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"{chat['title'][:30]}.md"
    )

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    # If this is a ChatGPT import, convert it first
    if chat_id.startswith('chatgpt_'):
        new_chat_id = convert_chatgpt_to_ollama(chat_id)
        chat_id = new_chat_id
    
    chats = load_ollama_chats()
    
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
    save_chats_direct(chats)
    
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
            system_prompt = """You are an autonomous AI assistant working on a task. For each response:
1. Evaluate what has been accomplished so far
2. Identify the next logical step
3. Execute that step thoroughly
4. State clearly if the task is COMPLETE or if more work is needed

Think step-by-step and work systematically. Break complex tasks into manageable pieces.
After each step, briefly state what you did and what comes next."""
            
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
        
        # Start timing
        start_time = time.time()
        
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
        
        # End timing
        end_time = time.time()
        generation_time = end_time - start_time
        
        if not response:
            response = "Error: No response from model. Make sure Ollama is running and the model is installed."
        
        # Calculate tokens per second (approximate)
        response_words = len(response.split())
        tokens_estimate = response_words * 1.3  # Rough estimate
        tokens_per_second = tokens_estimate / generation_time if generation_time > 0 else 0
        
        # Record generation stats
        if settings.get("track_generation_stats", True):
            gen_stats = record_generation(
                chat_id=chat_id,
                model=selected_model,
                prompt_length=len(full_prompt),
                response_length=len(response),
                duration=generation_time,
                tokens_per_second=tokens_per_second
            )
        else:
            gen_stats = None
        
        # Add assistant response
        chats[chat_id]["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().timestamp(),
            "model": selected_model,
            "auto_thinking": auto_thinking,
            "generation_time": round(generation_time, 2),
            "tokens_per_second": round(tokens_per_second, 2)
        })
        chats[chat_id]["updated"] = datetime.now().timestamp()
        save_chats_direct(chats)
        
        return jsonify({
            "response": response,
            "chat": chats[chat_id],
            "model": selected_model,
            "auto_thinking": auto_thinking,
            "generation_time": round(generation_time, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "stats": gen_stats
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Request timeout - try a simpler prompt"}), 408
    except FileNotFoundError:
        return jsonify({"error": "Ollama not found. Make sure it's installed and running."}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get generation statistics."""
    return jsonify(get_generation_stats_summary())

@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    """Reset generation statistics."""
    save_stats({"generations": []})
    return jsonify({"success": True})

@app.route('/api/chatgpt/import', methods=['POST'])
def import_chatgpt():
    """Manually trigger ChatGPT import/index refresh."""
    data = request.json or {}
    file_path = data.get('path')
    
    if file_path:
        # Import specific file
        try:
            index = build_chatgpt_index(Path(file_path))
            
            # Merge with existing index
            existing = load_chatgpt_index()
            existing.update(index)
            save_chatgpt_index(existing)
            
            return jsonify({
                "success": True,
                "imported": len(index),
                "message": f"Indexed {len(index)} ChatGPT conversations"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Refresh entire index
        try:
            index = refresh_chatgpt_index()
            return jsonify({
                "success": True,
                "imported": len(index),
                "message": f"Indexed {len(index)} ChatGPT conversations"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/chatgpt/cache/clear', methods=['POST'])
def clear_chatgpt_cache():
    """Clear the in-memory ChatGPT cache to free up memory."""
    global _chatgpt_cache
    cache_size = len(_chatgpt_cache)
    _chatgpt_cache.clear()
    return jsonify({
        "success": True,
        "cleared": cache_size,
        "message": f"Cleared {cache_size} cached conversations"
    })

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
    print("🚀 Ollama Chat Server Starting...")
    print("=" * 60)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🌐 Server: http://localhost:5001")
    print("\n✨ Features:")
    print("  • Smart Model Switching - Auto-selects best model")
    print("  • Auto Mode Iterations - Autonomous task completion")
    print("  • ChatGPT Import - Load & interact with ChatGPT conversations")
    print("  • Lazy Loading - Efficient memory usage for large chat histories")
    print("  • Generation Stats - Track response times and performance")
    print("  • File Upload Support - Analyze any text file")
    print("  • LaTeX Rendering - Beautiful math equations")
    print("=" * 60)
    
    # Build ChatGPT index on startup
    print("\n🔍 Scanning for ChatGPT exports...")
    chatgpt_index = refresh_chatgpt_index()
    
    if chatgpt_index:
        print(f"✓ Found {len(chatgpt_index)} ChatGPT conversations")
        print("💡 Full conversations load on-demand for better performance")
    else:
        print("ℹ No ChatGPT exports found")
        print(f"   Place conversations.json in: {CHATGPT_EXPORT_PATH}")
    
    # Show Ollama stats
    ollama_chats = load_ollama_chats()
    print(f"\n📝 Ollama chats: {len(ollama_chats)}")
    
    # Show stats if available
    stats = get_generation_stats_summary()
    if stats['total_generations'] > 0:
        print(f"\n📊 Generation Stats:")
        print(f"   • Total generations: {stats['total_generations']}")
        print(f"   • Average duration: {stats['average_duration']}s")
        if stats['by_model']:
            print(f"   • Models used: {', '.join(stats['by_model'].keys())}")
    
    print("\n💡 Make sure Ollama is running:")
    print("   ollama serve")
    print("\n📦 Install models:")
    print("   ollama pull llama3.2")
    print("   ollama pull codellama")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, port=5001, threaded=True)
