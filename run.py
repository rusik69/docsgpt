from llama_index import StorageContext, load_index_from_storage
import gradio as gr

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()


def chatbot(input_text):
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, 
                        label="Enter your text"),
                     outputs="text",
                     title="docs-trained AI Chatbot")
iface.launch(share=True)
