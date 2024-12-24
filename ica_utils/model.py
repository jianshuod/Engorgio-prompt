import os

import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import OPTForCausalLM, GPTNeoXForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import PeftModel
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed

from functools import partialmethod

root_path = '/PATH/TO/models/hf_models'

def fetch_model_size(name):
    if "3B" in name or "3b" in name: return "3b"
    elif "7B" in name or "7b" in name: return "7b"
    elif "12B" in name or "12b" in name: return "12b"
    elif "13B" in name or "13b" in name: return "13b"

nm2_local_cvt = {
    "llama/7B":"llama/hf",
    "alpaca": os.path.join(root_path, "alpaca/hf"),
    "chatglm":"THUDM/chatglm-6b",
    "dolly/7b":f"databricks/dolly-v2-7B"
}

'''
[StableLM]
    1. Model Info
        Parameters	Hidden Size	Layers	Heads	Sequence Length
        3B	        4096	16	        32	        4096
        7B	        6144	16	        48	        4096
    2. Input Format (StableLM Tuned should be used with prompts formatted to <|SYSTEM|>...<|USER|>...<|ASSISTANT|>... The system prompt is)
        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """
        prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

[Dolly-v2]
    1.Input Format
        from langchain import PromptTemplate, LLMChain
        from langchain.llms import HuggingFacePipeline

        # template for an instrution with no input
        prompt = PromptTemplate(
            input_variables=["instruction"],
            template="{instruction}")

        # template for an instruction with input
        prompt_with_context = PromptTemplate(
            input_variables=["instruction", "context"],
            template="{instruction}\n\nInput:\n{context}")

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

        llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
        llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
[oasst]
    1. input format
        Prompting
        Two special tokens are used to mark the beginning of user and assistant turns: <|prompter|> and <|assistant|>. Each turn ends with a <|endoftext|> token.

        Input prompt example:

        <|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>

        The input ends with the <|assistant|> token to signal that the model should start generating the assistant reply.    

'''

class StableLMStopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False



def get_ft_models(model_name, args):
    if model_name == "stablelm-7b":
        tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
        model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
        model.generate = partialmethod(model.generate, stopping_criteria=StoppingCriteriaList([StableLMStopOnTokens()]))
    elif model_name.startswith("dolly"):
        remote_model_name = f"databricks/dolly-v2-{fetch_model_size(model_name)}"
        tokenizer = AutoTokenizer.from_pretrained(remote_model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(remote_model_name, device_map="auto", load_in_8bit=args.load_in_8bit)
    elif model_name == 'chatglm':
        local_path = '/PATH/TO/ICA/chatglm'
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(local_path, trust_remote_code=True).half().cuda()
    elif "oasst" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
        model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", device_map='auto', load_in_8bit=args.load_in_8bit)
    elif model_name.startswith("vicuna"): # in local
        local_path = os.path.join(root_path,"FastChat/weights/7b")
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(local_path).cuda()
    elif model_name == "alpaca":  # in local
        local_path = os.path.join(root_path, "alpaca/hf")
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(local_path, device_map='auto', load_in_8bit=args.load_in_8bit)
    elif model_name == 'mosaic':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # a little special
        model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-7b', trust_remote_code=True).cuda()
    elif model_name == 'mosaic-instruct':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # a little special
        model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-7b-instruct', trust_remote_code=True).cuda()
    elif model_name == 'pythia':
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped", revision="step143000")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped", revision="step143000").cuda()
    elif model_name == 'wizardlm':
        local_path = '/PATH/TO/ICA/wizardlm'
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True).cuda()
        # unwind broken decapoda-research config
        # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        # model.config.bos_token_id = 1
        # model.config.eos_token_id = 2

    elif model_name == 'nous':
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/gpt4-x-vicuna-13b")
        model = LlamaForCausalLM.from_pretrained('NousResearch/gpt4-x-vicuna-13b').cuda()
    elif model_name == 'gptj':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    elif model_name == 'koala':
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/koala-7B-HF")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/koala-7B-HF").cuda()
    elif model_name == 'stablevicuna':
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/stable-vicuna-13B-GPTQ")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/stable-vicuna-13B-GPTQ").cuda()
    elif model_name == 'guanaco':
        local_path = '/PATH/TO/ICA/guanaco'
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForCausalLM.from_pretrained(local_path).cuda()

    # model.cuda()
    return tokenizer, model

def get_model(model_name, args):
    if model_name.startswith('gpt2'):
        local_path = '/PATH/TO/ICA/gpt2-large'
        tokenizer = GPT2Tokenizer.from_pretrained(local_path)
        model = GPT2LMHeadModel.from_pretrained(local_path).cuda()
    elif model_name == 'microsoft/DialoGPT-small':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'facebook/opt-125m':
        local_path = '/PATH/TO/ICA/opt-125m'
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = OPTForCausalLM.from_pretrained(local_path).cuda()
    elif model_name == 'facebook/opt-1.3b':
        local_path = '/PATH/TO/ICA/opt-1.3b'
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = OPTForCausalLM.from_pretrained(local_path).cuda()
    elif model_name.startswith('bigscience/bloom'):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name == 'llama/7B':
        tokenizer = LlamaTokenizer.from_pretrained("/PATH/TO/ICA/llama/hf")
        if args.load_in_8bit:
            model = LlamaForCausalLM.from_pretrained("/PATH/TO/ICA/llama/hf", device_map="auto", load_in_8bit=True)
        else:
            model = LlamaForCausalLM.from_pretrained("/PATH/TO/ICA/llama/hf")
            model.cuda()
    elif model_name == 'llama/30B':
        local_path = "/PATH/TO/models/hf_llama"
        tokenizer = LlamaTokenizer.from_pretrained(local_path)
        if args.load_in_8bit:
            model = LlamaForCausalLM.from_pretrained(local_path, device_map="auto", load_in_8bit=True)
        else:
            model = LlamaForCausalLM.from_pretrained(local_path).cuda()
    elif model_name == 'tloen/alpaca-lora-7b':
        base_model = 'decapoda-research/llama-7b-hf'
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        model = PeftModel.from_pretrained(
                model,
                model_name,
                torch_dtype=torch.float16,
            )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    elif model_name == 'project-baize/baize-lora-7B':
        base_model='huggyllama/llama-7b'
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            torch_dtype=torch.float16,
        )
    else:
        # tokenizer, model = get_ft_models(model_name, args)

        try:
            tokenizer, model = get_ft_models(model_name, args)
        except:
            raise ValueError(f"No model: {model_name}")
    
    return tokenizer, model


def load_deepspeed(model, args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    if args.model in nm2_local_cvt.keys():
        config = AutoConfig.from_pretrained(nm2_local_cvt[args.model])
        model_hidden_size = config.hidden_size
    else:
        config = AutoConfig.from_pretrained(args.model)
        model_hidden_size = config.d_model
    train_batch_size = 1 * world_size
    # ds_config = {
    #     "fp16": {
    #         "enabled": False
    #     },
    #     "bf16": {
    #         "enabled": False
    #     },
    #     "zero_optimization": {
    #         "stage": 3,
    #         "offload_param": {
    #             "device": "cpu",
    #             "pin_memory": True
    #         },
    #         "overlap_comm": True,
    #         "contiguous_gradients": True,
    #         "reduce_bucket_size": model_hidden_size * model_hidden_size,
    #         "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
    #         "stage3_param_persistence_threshold": 10 * model_hidden_size
    #     },
    #     "steps_per_print": 2000,
    #     "train_batch_size": train_batch_size,
    #     "train_micro_batch_size_per_gpu": 1,
    #     "wall_clock_breakdown": False,
    #     # "partition_activations":True
    # }
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,

            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 10 * model_hidden_size,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        # "gradient_accumulation_steps": "auto",
        # "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
    }
    dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
    model.gradient_checkpointing_enable()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    # ds_engine.module.eval()  # inference
    ds_engine.module.train()  # train

    return dschf, ds_engine.module

#
def get_vocab_size(model_name):
    if model_name.startswith('gpt2') or model_name == 'microsoft/DialoGPT-small':
        vocab_size = 50257
    elif model_name.startswith('facebook/opt'):
        vocab_size = 50272
    elif model_name.startswith('bigscience/bloom'):
        vocab_size = 250880
    elif model_name.startswith('llama') or model_name == 'alpaca':
        vocab_size = 32000
    else:
        raise ValueError(f"No model: {model_name}")
    
    return vocab_size