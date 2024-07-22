import gradio as gr
import pandas as pd
from transformers import pipeline
from load_models import models_and_tokenizers, models_checkpoints
import spaces

choice = {"ModelA": "", "ModelB": ""}

dff = pd.read_csv("models.csv")
dff.to_html("tab.html")

def refreshfn() -> gr.HTML:
    df = pd.read_csv("models.csv")
    df.to_html("tab.html")
    f = open("tab.html")
    content = f.read()
    f.close()
    t = gr.HTML(content)
    return t

def rewrite_csv_ordered_by_winning_rate(csv_path):
    # Read the input CSV
    df = pd.read_csv(csv_path)
    
    # Sort the DataFrame by WINNING_RATE in descending order
    df_sorted = df.sort_values(by="WINNING_RATE", ascending=False)
    
    # Save the sorted DataFrame to a new CSV file
    df_sorted.to_csv(csv_path, index=False)

@spaces.GPU(duration=200)
def run_inference(pipeline, prompt):
    response = pipeline(prompt)
    bot_message = response[0]["generated_text"]
    return bot_message

def modelA_button():
    global choice
    df = pd.read_csv("models.csv")
    df.loc[df["MODEL"] == choice["ModelA"], "MATCHES_WON"] += 1
    df.loc[df["MODEL"] == choice["ModelA"], "WINNING_RATE"] = df.loc[df["MODEL"] == choice["ModelA"], "MATCHES_WON"]/df.loc[df["MODEL"] == choice["ModelA"], "MATCHES_PLAYED"]  
    df.to_csv("models.csv", index=False)
    rewrite_csv_ordered_by_winning_rate("models.csv")

def modelB_button():
    global choice
    df = pd.read_csv("models.csv")
    df.loc[df["MODEL"] == choice["ModelB"], "MATCHES_WON"] += 1
    df.loc[df["MODEL"] == choice["ModelB"], "WINNING_RATE"] = df.loc[df["MODEL"] == choice["ModelB"], "MATCHES_WON"]/df.loc[df["MODEL"] == choice["ModelB"], "MATCHES_PLAYED"]  
    df.to_csv("models.csv", index=False)
    rewrite_csv_ordered_by_winning_rate("models.csv")
      
import time

def replyA(prompt, history, modelA):
    global choice
    choice["ModelA"] = modelA
    df = pd.read_csv("models.csv")
    df.loc[df["MODEL"] == modelA, "MATCHES_PLAYED"] += 1
    df.to_csv("models.csv", index=False)
    pipeA = pipeline("text-generation", model=models_and_tokenizers[modelA][0], tokenizer=models_and_tokenizers[modelA][1], max_new_tokens=512, repetition_penalty=1.5, temperature=0.5, device_map="cuda:0")
    responseA = run_inference(pipeA, prompt)    
    r = ''
    for c in responseA:
        r+=c
        time.sleep(0.0001)
        yield r

def replyB(prompt, history, modelB):
    global choice
    choice["ModelB"] = modelB
    df = pd.read_csv("models.csv")
    df.loc[df["MODEL"] == modelB, "MATCHES_PLAYED"] += 1
    df.to_csv("models.csv", index=False)
    pipeB = pipeline("text-generation", model=models_and_tokenizers[modelB][0], tokenizer=models_and_tokenizers[modelB][1], max_new_tokens=512, repetition_penalty=1.5, temperature=0.5, device_map="cuda:0")
    responseB = run_inference(pipeB, prompt)    
    r = ''
    for c in responseB:
        r+=c
        time.sleep(0.0001)
        yield r

modelAchoice = gr.Dropdown(models_checkpoints, label="Model A")
modelBchoice = gr.Dropdown(models_checkpoints, label="Model B")

with gr.Blocks() as demo2:
    f = open("tab.html")
    content = f.read()
    f.close()
    t = gr.HTML(content)
    btn = gr.Button("Refresh")
    btn.click(fn=refreshfn, inputs=None, outputs=t)


accrdnA = gr.Accordion(label="Choose model A", open=False)
accrdnB = gr.Accordion(label="Choose model B", open=False)

chtbA = gr.Chatbot(label="Chat with Model A", height=150)
chtbB = gr.Chatbot(label="Chat with Model B", height=150)

with gr.Blocks() as demo1:
    with gr.Column():
        gr.HTML("""<h1 align='center'>SmolLM Arena</h1>
<h2 align='center'>Cast your vote to choose the best Small Language Model (100M-1.7B)!🚀</h2>
<h3 align='center'>[<a href="https://github.com/AstraBert/smollm-arena">GitHub</a>] [<a href="https://github.com/AstraBert/smollm-arena?tab=readme-ov-file#usage">Usage Guide</a>]""")
        gr.ChatInterface(fn=replyA, chatbot=chtbA, additional_inputs=modelAchoice, additional_inputs_accordion=accrdnA, submit_btn="Submit to Model A")
        gr.ChatInterface(fn=replyB, chatbot=chtbB, additional_inputs=modelBchoice, additional_inputs_accordion=accrdnB, submit_btn="Submit to Model B")
    with gr.Column():
        btnA = gr.Button("Vote for Model A!") 
        btnB = gr.Button("Vote for Model B!")
        btnA.click(modelA_button, inputs=None, outputs=None) 
        btnB.click(modelB_button, inputs=None, outputs=None) 

demo = gr.TabbedInterface([demo1, demo2], ["Chat Arena", "Leaderboard"])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)