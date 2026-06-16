import gradio as gr
from main import chat, process_document

CSS = """
footer {
    display: none !important;
}

/* =========================
   APP LAYOUT
========================= */
.gradio-container {
    max-width: 1400px !important;
    width: 95vw !important;
    margin: auto !important;
    padding-top: 10px !important;
}

/* Remove default borders/shadows */
.block {
    border: none !important;
    box-shadow: none !important;
}

/* =========================
   HEADER
========================= */
#header {
    padding: 10px 0px;
}

#title {
    font-size: 30px;
    font-weight: 700;
}

/* =========================
   CHAT AREA
========================= */
.message {
    border-radius: 18px !important;
    padding: 12px !important;
}

/* Markdown text */
.message p,
.message li,
.message span,
.message div,
.message pre,
.message code {
    line-height: 1.6 !important;
}

/* Code blocks */
.message pre {
    border-radius: 12px !important;
    padding: 12px !important;
    overflow-x: auto !important;
}

/* =========================
   INPUT BOX
========================= */
textarea {
    border-radius: 24px !important;
    font-size: 16px !important;
    padding: 12px !important;
}

/* Placeholder */
textarea::placeholder {
    opacity: 0.8;
}

/* =========================
   UPLOAD BUTTON
========================= */
#upload {
    max-width: 70px !important;
}

#upload button {
    min-width: 55px !important;
    border-radius: 18px !important;
}

/* =========================
   BUTTONS
========================= */
button {
    border-radius: 18px !important;
}

/* =========================
   LIGHT MODE
========================= */
body.light-mode {
    background: #ffffff !important;
}

body.light-mode .gradio-container {
    background: #ffffff !important;
}

body.light-mode .message {
    background: #f7f7f8 !important;
    color: #111111 !important;
}

body.light-mode .message * {
    color: #111111 !important;
}

body.light-mode textarea,
body.light-mode input {
    background: #ffffff !important;
    color: #111111 !important;
}

/* =========================
   DARK MODE
========================= */
body.dark-mode {
    background: #212121 !important;
}

body.dark-mode .gradio-container {
    background: #212121 !important;
}

/* Chat bubbles */
body.dark-mode .message {
    background: #2f2f2f !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
    border-radius: 18px !important;
}

/* EVERYTHING inside bubbles */
body.dark-mode .message * {
    color: #ffffff !important;
}

/* Markdown elements */
body.dark-mode .message p,
body.dark-mode .message li,
body.dark-mode .message span,
body.dark-mode .message div,
body.dark-mode .message pre,
body.dark-mode .message code,
body.dark-mode .message strong,
body.dark-mode .message em {
    color: #ffffff !important;
}

/* Code blocks */
body.dark-mode .message pre {
    background: #1e1e1e !important;
    color: #ffffff !important;
}

/* Input area */
body.dark-mode textarea,
body.dark-mode input {
    background: #303030 !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
}

/* Placeholder */
body.dark-mode textarea::placeholder {
    color: #b0b0b0 !important;
}

/* Buttons */
body.dark-mode button {
    background: #303030 !important;
    color: #ffffff !important;
    border: 1px solid #404040 !important;
}

/* Upload component */
body.dark-mode #upload button {
    background: #303030 !important;
    color: white !important;
}

/* Chatbot container */
body.dark-mode .chatbot {
    color: #ffffff !important;
}
"""

THEME_JS = """
(currentTheme) => {

    if (currentTheme === "light") {

        document.body.classList.remove(
            "light-mode"
        );

        document.body.classList.add(
            "dark-mode"
        );

        return ["dark", "☀️"];
    }

    document.body.classList.remove(
        "dark-mode"
    );

    document.body.classList.add(
        "light-mode"
    );

    return ["light", "🌙"];
}
"""


with gr.Blocks(
    title="Knowledge Worker"
) as demo:

    history = gr.State([])
    theme_state = gr.State("light")

    # ==========================
    # Header
    # ==========================
    with gr.Row(
        elem_id="header"
    ):

        gr.Markdown(
            """
            <div id="title">
            Knowledge Worker
            </div>
            """
        )

        theme_btn = gr.Button(
            "🌙",
            min_width=60
        )

    # ==========================
    # Chat Area
    # ==========================
    chatbot = gr.Chatbot(
        height="80vh",
        avatar_images=(None, None)
    )

    # ==========================
    # Input Area
    # ==========================
    with gr.Row():

        upload_btn = gr.UploadButton(
            "📎",
            file_count="multiple",
            file_types=[
                ".pdf",
                ".docx"
            ],
            elem_id="upload",
            scale=1
        )

        msg = gr.Textbox(
            placeholder="Message Knowledge Worker...",
            container=False,
            scale=10
        )

        send_btn = gr.Button(
            "➤",
            variant="primary",
            min_width=60,
            scale=1
        )

    # ==========================
    # Upload Documents
    # ==========================
    upload_btn.upload(
        fn=process_document,
        inputs=upload_btn,
        outputs=None
    )

    # ==========================
    # Send Message
    # ==========================
    msg.submit(
        fn=chat,
        inputs=[
            msg,
            history
        ],
        outputs=[
            chatbot,
            history,
            msg
        ]
    )

    send_btn.click(
        fn=chat,
        inputs=[
            msg,
            history
        ],
        outputs=[
            chatbot,
            history,
            msg
        ]
    )

    # ==========================
    # Theme Toggle
    # ==========================
    theme_btn.click(
        fn=None,
        inputs=theme_state,
        outputs=[
            theme_state,
            theme_btn
        ],
        js=THEME_JS
    )

    # ==========================
    # Initialize Theme
    # ==========================
    demo.load(
        None,
        js="""
        () => {
            document.body.classList.add(
                "light-mode"
            );
        }
        """
    )

demo.launch(
    theme=gr.themes.Soft(),
    css=CSS
)