from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
import gc
from datetime import datetime
import re
from loguru import logger
import warnings


gc.collect()
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def get_model():
    bnb_config = BitsAndBytesConfig(  
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= True,
        llm_int8_enable_fp32_cpu_offload= True
    )

    model = AutoModelForCausalLM.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.2',
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            )
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
    return model, tokenizer


def model_seq_gen(model, tokenizer, eval_prompt): 
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
        start = datetime.now()
        sequences = pipe(
            f'{eval_prompt}' ,
            do_sample=True,
            max_new_tokens=1024, 
            temperature=0.7, 
            top_p=0.95
        )
        extracted_title = re.sub(r'[\'"]', '', sequences[0]['generated_text'].split("[/INST]")[1])
        stop = datetime.now()
        time_taken = stop-start
        print(f"Execution Time : {time_taken}")
        return extracted_title


def formatting_func(example):
    text = f"<s>[INST]Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.{example['instruction']}\n\n### Input:{example['input']}[/INST]"
    return text


def main(model, tokenizer, prompt):

    torch.cuda.empty_cache()

    result = model_seq_gen(model, tokenizer, prompt)
    print(result)
    



if __name__ == '__main__':
    model, tokenizer = get_model()
    try:
        while True:
            print('Type in your instruction: ')
            instruction = input()
            print('Type in you input: ')
            input_ = input()
            example = {
                    'instruction': instruction,
                    'input': input_
                    }
            eval_prompt = formatting_func(example)
            main(model, tokenizer, eval_prompt)
    except KeyboardInterrupt:
        exit()
