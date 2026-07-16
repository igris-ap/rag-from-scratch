"""
app.py — Gradio chat interface for the RAG system.
Compatible with Gradio 6.x+

Run with:
    python3 app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import os
import shutil
from pathlib import Path

from llm import chat
from chunker import convert_all_pdfs, index_all_documents
from vector_store import setup_db, clear_chunks, store_children
from rag import retrieve
from query_intelligence import analyze_query, summarize_conversation
from main import process_turn, index_documents, MAX_HISTORY


# ---------------------------------------------------------------------------
# Indexing helper
# ---------------------------------------------------------------------------

def reindex():
    """Re-run the full indexing pipeline. Returns a status string."""
    convert_all_pdfs()
    children = index_all_documents()

    if not children:
        return "No documents found. Upload PDFs using the panel on the left."

    clear_chunks()
    store_children(children)
    return f"✓ Indexed {len(children)} chunks. Ready to chat."


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def respond(
    message: str,
    chat_history: list,
    show_analysis: bool,
    conversation_state: list,
):
    """
    Called by Gradio on every user message.

    Returns:
        ("", updated_chat_history, updated_conversation_state)
    """
    if not message.strip():
        return "", chat_history, conversation_state

    # Query analysis for verbose mode
    recent   = conversation_state[-MAX_HISTORY:]
    summary  = summarize_conversation(recent)
    analysis = analyze_query(message, summary)

    # Process the turn through the full pipeline — verbose mirrors the
    # "Show query analysis" checkbox, so checking it also prints the
    # agent's tool-selection / rerank / reflection / critique steps to
    # the terminal running app.py.
    reply, conversation_state = process_turn(message, conversation_state, verbose=show_analysis)

    # Prepend analysis info if verbose mode is on
    if show_analysis:
        analysis_md = (
            f"> **Query analysis**\n"
            f"> - Clear: `{analysis['is_clear']}`\n"
            f"> - Sub-questions: `{analysis['questions']}`\n\n"
        )
        if summary:
            analysis_md += f"> - Summary: _{summary[:120]}_\n\n"
        display_reply = analysis_md + reply
    else:
        display_reply = reply

    # Gradio 6.12 requires dict format
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": display_reply})

    return "", chat_history, conversation_state


# ---------------------------------------------------------------------------
# File upload handler
# ---------------------------------------------------------------------------

def upload_and_index(files):
    """Copy uploaded PDFs to docs/ and re-index."""
    if not files:
        return "No files uploaded."

    os.makedirs("docs", exist_ok=True)
    names = []

    for f in files:
        dest = Path("docs") / Path(f.name).name
        shutil.copy(f.name, dest)
        names.append(dest.name)

    status = f"Uploaded: {', '.join(names)}\nRe-indexing..."
    result = reindex()
    return status + "\n" + result


# ---------------------------------------------------------------------------
# Clear conversation
# ---------------------------------------------------------------------------

def clear_conversation():
    return [], []   # empty chat history, empty conversation state


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="RAG from Scratch") as app:

        gr.Markdown(
            """
            # 🔍 RAG from Scratch
            **No LangChain. No LangGraph. Just Python.**
            Upload your PDFs on the left, then ask questions on the right.
            """
        )

        # Per-session conversation state
        conversation_state = gr.State([])

        with gr.Row():

            # ---- Left sidebar ----
            with gr.Column(scale=1, min_width=260):

                gr.Markdown("### 📄 Documents")

                file_upload = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple",
                )

                reindex_btn = gr.Button("🔄 Re-index", variant="secondary", size="sm")

                status_box = gr.Textbox(
                    label="Status",
                    value="Ready.",
                    interactive=False,
                    lines=4,
                )

                gr.Markdown("### ⚙️ Options")

                show_analysis = gr.Checkbox(
                    label="Show query analysis",
                    value=False,
                    info="Shows how queries are rewritten and split",
                )

                clear_btn = gr.Button("🗑️ Clear conversation", variant="secondary", size="sm")

                gr.Markdown(
                    """
                    ### ℹ️ How it works
                    1. Upload PDFs → Re-index
                    2. Ask any question
                    3. System retrieves relevant chunks and answers

                    **Tips**
                    - Multi-part questions get split automatically
                    - Vague questions trigger a clarification request
                    - Enable query analysis to see what's happening
                    """
                )

            # ---- Right: chat ----
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="Chat",
                    elem_id="chatbot",
                    show_label=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        show_label=False,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send ↵", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "What is the main topic of the uploaded documents?",
                        "Summarize the key points.",
                        "What are the main components described?",
                    ],
                    inputs=msg_input,
                )

        # ---- Wire up events ----

        # Send on button click
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, show_analysis, conversation_state],
            outputs=[msg_input, chatbot, conversation_state],
        )

        # Send on Enter key
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, show_analysis, conversation_state],
            outputs=[msg_input, chatbot, conversation_state],
        )

        # Upload → auto index
        file_upload.upload(
            fn=upload_and_index,
            inputs=[file_upload],
            outputs=[status_box],
        )

        # Manual reindex
        reindex_btn.click(
            fn=reindex,
            outputs=[status_box],
        )

        # Clear
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, conversation_state],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Setting up database...")
    setup_db()

    print("Indexing existing documents...")
    index_documents()

    print("\nStarting Gradio app...")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css="#chatbot { height: 520px; }",
    )
