import streamlit as st
import requests
import json
from collections import deque


def _clean_raw_bytes(line: bytes):
    """
    Cleans the raw bytes from the server and converts to OpenAI format.

    Args:
        line (bytes): The raw bytes from the server

    Returns:
        dict or None: Parsed JSON response in OpenAI format, or None if parsing fails
    """
    try:
        # Handle OpenAI format
        if line.startswith(b"data: "):
            json_str = line.decode("utf-8")[6:]  # Remove 'data: ' prefix
            return json.loads(json_str)
        # Handle Ollama format
        else:
            return json.loads(line.decode("utf-8"))
    except Exception as e:
        # logger.warning(f"Failed to parse server response: {e}")
        return None


def _process_chunk(line: dict):
    """
    Processes a single line of text from the LLM server.

    Args:
        line (dict): The line of text from the LLM server.
    """
    if not line:
        return None

    try:
        # Handle OpenAI format
        if "choices" in line:
            content = line.get("choices", [{}])[0].get("delta", {}).get("content")
            return content if content else None
        # Handle Ollama format
        else:
            content = line.get("message", {}).get("content")
            return content if content else None
    except Exception as e:
        # logger.error(f"Error processing chunk: {e}")
        return None


class ThinkParser:
    def __init__(self):
        self.buffer = ""
        self.in_think = False
        self.open_think_id = None

    def process(self, text):
        self.buffer += text
        parts = []
        while True:
            if not self.in_think:
                start = self.buffer.find("<think>")
                if start == -1:
                    if self.buffer:
                        parts.append(("text", self.buffer))
                        self.buffer = ""
                    break
                else:
                    if start > 0:
                        parts.append(("text", self.buffer[:start]))
                    self.buffer = self.buffer[start + 7 :]
                    self.in_think = True
                    self.open_think_id = f"think-{len(parts)}"
                    parts.append(("think_open", self.open_think_id))
            else:
                end = self.buffer.find("</think>")
                if end == -1:
                    content = self.buffer
                    parts.append(("think_update", (self.open_think_id, content)))
                    self.buffer = ""
                    break
                else:
                    content = self.buffer[:end]
                    parts.append(("think_update", (self.open_think_id, content)))
                    parts.append(("think_close", self.open_think_id))
                    self.buffer = self.buffer[end + 8 :]
                    self.in_think = False
                    self.open_think_id = None
        return parts

st.set_page_config(page_title="DeepSeek-R1 Demo")

st.markdown(
    """
<style>
.thinking {
    border-left: 3px solid #4e79a7;
    margin: 1em 0;
    padding: 0.5em 1em;
    background: #f8f9fa;
    border-radius: 4px;
    position: relative;
}
.thinking[data-open="true"] {
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.thinking summary {
    font-weight: 500;
    color: #2c3e50;
    cursor: pointer;
    outline: none;
}
.thinking-content {
    margin-top: 0.5em;
    color: #34495e;
    white-space: pre-wrap;
}
.streaming-cursor {
    display: inline-block;
    width: 2px;
    background: #4e79a7;
    margin-left: 2px;
    animation: blink 1s step-end infinite;
}
@keyframes blink {
    50% { opacity: 0 }
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("DeepSeek-R1 Demo")
st.caption("Since the official app sends your personal data to the CCP bruh")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "user",
            "content": "You are an AI programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan for what to build in pseudocode, written out in great detail. Then, output the code in a single code block. Minimize any other prose.",
            "is_system": True,
        }
    ]
if "thinking" not in st.session_state:
    st.session_state.thinking = {}

for msg in st.session_state.messages:
    if not msg.get("is_system"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        final_output = []
        parser = ThinkParser()
        thinking_order = deque()

        with requests.post(
            "https://nearby-good-parakeet.ngrok-free.app/api/chat",
            headers={
                "Authorization": "Bearer test",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "stream": True,
                "messages": [
                    m for m in st.session_state.messages if not m.get("is_system")
                ],
            },
            stream=True,
        ) as response:

            for line in response.iter_lines():
                if line:
                    cleaned = _clean_raw_bytes(line)
                    chunk = _process_chunk(cleaned)

                    if chunk:
                        parts = parser.process(chunk)
                        update_needed = False

                        for part_type, content in parts:
                            if part_type == "text":
                                final_output.append(content)
                                update_needed = True
                            elif part_type == "think_open":
                                thinking_id = content
                                st.session_state.thinking[thinking_id] = {
                                    "content": "",
                                    "open": True,
                                }
                                thinking_order.append(thinking_id)
                                update_needed = True
                            elif part_type == "think_update":
                                thinking_id, delta = content
                                st.session_state.thinking[thinking_id][
                                    "content"
                                ] += delta
                                update_needed = True
                            elif part_type == "think_close":
                                thinking_id = content
                                st.session_state.thinking[thinking_id]["open"] = False
                                update_needed = True

                        if update_needed:
                            # Build thinking sections HTML
                            thinking_html = []
                            for tid in thinking_order:
                                think = st.session_state.thinking[tid]
                                content = think["content"]
                                if think["open"]:
                                    content += '<span class="streaming-cursor"></span>'

                                thinking_html.append(
                                    f"""
                                <div class="thinking" data-open="{str(think['open']).lower()}">
                                    <details {'open' if think['open'] else ''}>
                                        <summary>🤔 Thinking Process</summary>
                                        <div class="thinking-content">{content}</div>
                                    </details>
                                </div>
                                """
                                )

                            # Build final display
                            display_content = "\n".join(thinking_html) + "".join(
                                final_output
                            )
                            if any(
                                t["open"] for t in st.session_state.thinking.values()
                            ):
                                display_content += (
                                    '<span class="streaming-cursor"></span>'
                                )

                            response_placeholder.markdown(
                                display_content, unsafe_allow_html=True
                            )

            # Final update to remove any remaining cursors
            display_content = (
                "\n".join(
                    f"""<div class="thinking" data-open="false">
                    <details>
                        <summary>🤔 Thinking Process</summary>
                        <div class="thinking-content">{st.session_state.thinking[t]['content']}</div>
                    </details>
                </div>"""
                    for t in thinking_order
                )
                + "".join(final_output)
            )

            response_placeholder.markdown(st.latex(display_content), unsafe_allow_html=True)
            st.session_state.thinking.clear()

    st.session_state.messages.append({"role": "assistant", "content": display_content})
