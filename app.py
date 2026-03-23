import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Blankyy/reasoning-math-model"
base_model_id = "unsloth/Qwen3-0.6B-Base-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype="auto", device_map="auto"
)
# Load your fine-tuned adapters on top
model = PeftModel.from_pretrained(base_model, model_id)


def respond(message):
    start = "<|im_start|>"
    end = "<|im_end|>"
    think_start = "<think>"
    think_end = "</think>"
    message = f"{start}user\n{message}\n{end}\n{start}assistant\n{think_start}\n"
    tokenized_message = tokenizer(
        message, add_special_tokens=False, return_tensors="pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {
        "input_ids": tokenized_message["input_ids"].to(device),
        "attention_mask": tokenized_message["attention_mask"].to(device),
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(end),
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=False
    )

    cot = ""

    if think_end in generated_text:
        cot, post_think = generated_text.split(think_end)
        extracted_answer = post_think.split(end)[0].strip()
    else:
        extracted_answer = generated_text.split(end)[0].strip()

    return extracted_answer, cot


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Chain of Thoughts Reasoning Demo
        A demo for Chain of Thoughts reasoning model trained on grade school math problems.
        Model used was fine-tuned on top of Qwen3-0.6B-Base-bnb-4bit using PEFT.
        The model is trained to generate a chain of thoughts before giving the final answer, which can be helpful for complex reasoning tasks.
        """
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type a message and press Enter...",
            label="Input",
            scale=3,
        )
    chain_of_thoughts = gr.Textbox(
        label="Chain of Thoughts",
        interactive=False,
        lines=4,
    )
    out = gr.Textbox(
        label="Output",
        interactive=False,
        lines=4,
    )
    clear = gr.Button("Clear", variant="secondary")

    msg.submit(respond, msg, [out, chain_of_thoughts])
    clear.click(lambda: ("", ""), None, [msg, out], queue=False)


demo.launch()
