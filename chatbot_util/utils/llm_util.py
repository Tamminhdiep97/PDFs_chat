from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, BitsAndBytesConfig
from transformers import pipeline
from loguru import logger
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline



class LLM(object):
    def __init__(self, config):
        self.get_model_pipeline(config)
        self.prompt = self.get_template()
        self.history = []

    def get_template(self):
        prompt = """<s>[INST]You are a helpful, respectful and honest AI assistant.
        Current conversation:{history}
        Human: {input}
        AI Assistant:"""
        return prompt

    def get_model_pipeline(self, config):
        bnb_config = BitsAndBytesConfig(  
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.float16,
            bnb_4bit_use_double_quant= True,
            llm_int8_enable_fp32_cpu_offload= True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm_tokenizer,
                trust_remote_code=True
            )
        self.pipe = pipeline(
                task='text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=0.1,
                top_p=0.15,
                top_k=500,
                max_new_tokens=1024,
                repetition_penalty=1.1
            )
        self.hf_llm = HuggingFacePipeline(pipeline=self.pipe)

    def modify_history(self, message, response):
        self.history.append([message, response])
            
    def get_response(self, message):
        history_transformer_format = self.history
        history_format = "".join(["".join(["\nHuman: "+item[0], "\nAI Assistant: "+item[1]])
                                  for item in history_transformer_format])   
        messages = self.prompt.format(history=history_format, input=message)
        messages += ' [/INST]'
        model_inputs = self.tokenizer([messages], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(self.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=0.7,
            num_beams=1
            # stopping_criteria=StoppingCriteriaList([stop])
            )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != '<':
                partial_message += new_token
                yield partial_message
        self.modify_history()
