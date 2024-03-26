from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from langchain.prompts.prompt import PromptTemplate
from loguru import logger

import config as conf
import function_wrapper

# def upload_file(filepath):
#     name = Path(filepath).name
#     return [gr.UploadButton(visible=False), gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)]
# 
# 
# def get_model():
#     bnb_config = BitsAndBytesConfig(  
#         load_in_4bit= True,
#         bnb_4bit_quant_type= "nf4",
#         bnb_4bit_compute_dtype= torch.float16,
#         bnb_4bit_use_double_quant= True,
#         llm_int8_enable_fp32_cpu_offload= True
#     )
# 
#     model = AutoModelForCausalLM.from_pretrained(
#             'mistralai/Mistral-7B-Instruct-v0.2',
#             quantization_config=bnb_config,
#             device_map="auto",
#             trust_remote_code=True,
#             )
#     tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
#     return model, tokenizer
# 
# 
# model, tokenizer = get_model()
# 
# 
# template = """<s>[INST]You are a helpful, respectful and honest AI assistant.
# Current conversation:{history}
# Human: {input}
# AI Assistant:"""
# 
# # text = """<s>[INST]Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# # ### Intructions:
# # {intruction}
# # ### Input:
# # {input}[/INST]"""
# # messages_template = [
# #     {"role": "system", "content": "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."}
# # ]
# 
# messages_template = []
# 
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [29, 0]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# def predict(message, history):
#     # logger.info(message)
#     logger.info(history)
#     history_transformer_format = history # + [[message, ""]]
#     # logger.info(history_transformer_format)
#     stop = StopOnTokens()
# 
#     # messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
#     #             for item in history_transformer_format])
#     history_format = "".join(["".join(["\nHuman: "+item[0], "\nAI Assistant: "+item[1]])
#                               for item in history_transformer_format])   
#     messages = template.format(history=history_format, input=message)
#     messages += ' [/INST]'
#     # logger.info(messages)
#     # history_format = messages_template.copy()
# 
#     # logger.info(history_format)
#     # for item in history_transformer_format:
#     #     history_format.append({'role': 'user', 'content:': item[0]})
#     #     history_format.append({'role': 'assistant', 'content:': item[1]})
#     # messages = history_format.copy()
#     # messages.append({'role': 'user', 'content': message})
#     logger.info(messages)
#     model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
#     streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
#     generate_kwargs = dict(
#         model_inputs,
#         streamer=streamer,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.95,
#         top_k=1000,
#         temperature=0.7,
#         num_beams=1
#         # stopping_criteria=StoppingCriteriaList([stop])
#         )
#     t = Thread(target=model.generate, kwargs=generate_kwargs)
#     t.start()
# 
#     partial_message = ""
#     for new_token in streamer:
#         if new_token != '<':
#             partial_message += new_token
#             yield partial_message

chatbot_function = function_wrapper.FunctionWrapper(conf)

def predict(message, history):
    answer = chatbot_function.answer_query(message)
    return answer

gr.ChatInterface(predict).launch(
        server_name='0.0.0.0',
        server_port=conf.ui_port
    )
