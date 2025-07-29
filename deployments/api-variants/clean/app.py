import gradio as gr

def hello_world(name):
    return f"Hello {name}! API is working!"

demo = gr.Interface(
    fn=hello_world,
    inputs="text",
    outputs="text",
    title="Depth Estimation API - Test"
)

demo.launch()