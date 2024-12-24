import time
import numpy as np
from typing import List

import torch
import torch.nn.functional as F

import transformers.deepspeed

import deepspeed

from .cost import get_cpu_reading
import os
import pynvml


from .model import get_model, load_deepspeed

def get_pred_length(model, trigger_tokens, tokenizer, device, args):
    batch_size = 8
    sample_time = 50
    query_time = sample_time // batch_size
    sum_time = 0
    sum_len = 0
    sum_max = 0
    cnt = 0
    for i in range(query_time + 1):
        bs = batch_size if batch_size * (i + 1) < sample_time else sample_time - (batch_size * i)
        if bs == 0: continue
        trigger_tokens_tensor = torch.tensor([trigger_tokens]).repeat(bs, 1)
        trigger_tokens_tensor = trigger_tokens_tensor.to(device)
        start_time = time.time()
        if args.top_k == 0:
            out = model.generate(trigger_tokens_tensor, do_sample=True, temperature=args.temperature, top_p=args.top_p, max_length=args.max_length)
        else:
            out = model.generate(trigger_tokens_tensor, do_sample=True, temperature=args.temperature, top_k=args.top_k, max_length=args.max_length)
        end_time = time.time()
        
        time_used = end_time - start_time
        sum_time += time_used

        for x in out:
            t = (x == tokenizer.pad_token_id).nonzero(as_tuple=True)[0] if tokenizer.pad_token_id is not None else (x == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if t.nelement() == 0 :
                cnt_len = args.max_length
            else:
                cnt_len = int(t[0]) + 1
            # print('--------------')
            # print(cnt, cnt_len)
            # print(tokenizer.decode(x, skip_special_tokens=True))
            # cnt += 1
            sum_len += cnt_len
            if cnt_len == args.max_length:
                sum_max += 1
    avg_time = sum_time / sample_time
    avg_len = sum_len / sample_time
    avg_rate = sum_max / sample_time

    return avg_time, avg_len, avg_rate

def cal_real_length(sample, tokenizer, max_length):
    x = sample
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t = (x == pad_token_id).nonzero(as_tuple=True)[0]
    cnt_len = min(x.size()[-1], max_length) if t.nelement() == 0 else int(t[0]) + 1
    return cnt_len

def cal_real_length_alpaca(sample, tokenizer, max_length, pad_token_id=-1):
    x = sample
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    t = (x[1:] == pad_token_id).nonzero(as_tuple=True)[0]
    cnt_len = min(x.size()[-1], max_length) if t.nelement() == 0 else int(t[0]) + 1

    return cnt_len

def eval_triggers(trigger_list, device, args, model_existing=None):
    torch.cuda.empty_cache()
    transformers.deepspeed.unset_hf_deepspeed_config()

    if not isinstance(trigger_list[0], List):
        trigger_list = [trigger_list]

    batch_size = args.bs
    sample_time = args.sample_time
    max_length = args.max_length
    model_name = args.model
    if model_existing is None:
        tokenizer, model = get_model(model_name, args)
    else:
        tokenizer, model = model_existing

    # if args.load_in_8bit:
    # ds_model = deepspeed.init_inference(
    #     model=model,      # Transformers models
    #     mp_size=1,        # Number of GPU
    #     dtype=torch.int8 if args.load_in_8bit else torch.float16, # dtype of the weights (fp16)
    #     replace_method="auto", # Lets DS autmatically identify the layer to replace
    #     replace_with_kernel_inject=True, # replace the model with the kernel injector
    # )
    
    trigger_info = []
    for trigger_tokens in trigger_list:
        input_length = len(trigger_tokens)
        remaining_samples = sample_time
        cnt, sum_max = 0, 0

        s_time = time.time()
        length_list = []
        while remaining_samples > 0:
            bs = min(remaining_samples, batch_size)
            remaining_samples -= bs
            cnt += 1

            trigger_tokens_tensor = torch.tensor([trigger_tokens]).repeat(bs, 1).to(device)
            out = model.generate(
                input_ids=trigger_tokens_tensor, 
                do_sample=True, 
                temperature=args.temperature, 
                max_length=max_length, 
                pad_token_id=tokenizer.pad_token_id,
            )

            for x in out:
                if args.model != 'tloen/alpaca-lora-7b':
                    cnt_len = cal_real_length(x, tokenizer, max_length)
                else:
                    cnt_len = cal_real_length_alpaca(x, tokenizer, max_length)
                length_list.append(cnt_len)
                if cnt_len == max_length: sum_max += 1
                print(tokenizer.decode(x))
                print(cnt_len, '-----------')
            print(f"Part {cnt}, Time Cost: {time.time() - s_time}")
        sum_time = time.time() - s_time
        avg_time = sum_time / sample_time
        avg_len = np.mean(length_list)
        std_len = np.std(length_list)
        avg_rate = sum_max / sample_time
        ratio = (avg_len - input_length) / input_length
        trigger_info.append((avg_time, avg_len, std_len, avg_rate, ratio))
        print(trigger_info)
    return trigger_info

def contrastive_eval(trigger_list, device, args, model_existing=None):
    torch.cuda.empty_cache()
    transformers.deepspeed.unset_hf_deepspeed_config()

    if not isinstance(trigger_list[0], List):
        trigger_list = [trigger_list]

    max_length = args.max_length
    model_name = args.model
    if model_existing is None:
        tokenizer, model = get_model(model_name, args)
    else:
        tokenizer, model = model_existing

    trigger_info = []
    for trigger_tokens in trigger_list:

        s_time = time.time()
        trigger_tokens_tensor = torch.tensor([trigger_tokens]).to(device)
        out = model.generate(
                input_ids=trigger_tokens_tensor, 
                temperature=args.temperature, 
                max_length=max_length, 
                pad_token_id=tokenizer.pad_token_id,
                top_k=args.top_k, 
                penalty_alpha=args.penalty_alpha
            )
        cnt_time = time.time() - s_time
        cnt_len = len(out[0])
        print(tokenizer.decode(out[0]))
        print(cnt_len, '-----------')
        trigger_info.append([cnt_time, cnt_len])
    print(trigger_info)


def eval_triggers_v2(trigger_list, device, args, model_existing=None, dev="gpu"):
    '''
        Not only running time, but also the power consumption is recorded.
    '''
    torch.cuda.empty_cache()
    transformers.deepspeed.unset_hf_deepspeed_config()

    if not isinstance(trigger_list[0], List):
        trigger_list = [trigger_list]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))

    batch_size = args.bs
    sample_time = args.sample_time
    max_length = args.max_length
    model_name = args.model
    if model_existing is None:
        tokenizer, model = get_model(model_name, args)
    else:
        tokenizer, model = model_existing
    
    trigger_info = []
    point_list = []
    for trigger_tokens in trigger_list:
        input_length = len(trigger_tokens)
        remaining_samples = sample_time
        cnt, sum_max = 0, 0
        
        length_list = []
        while remaining_samples > 0:

            bs = min(remaining_samples, batch_size)
            remaining_samples -= bs
            cnt += 1

            trigger_tokens_tensor = torch.tensor([trigger_tokens]).repeat(bs, 1).to(device)
            if dev == "gpu":
                before_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            else:
                before_energy = get_cpu_reading()
            before_time = time.time()
            out = model.generate(
                input_ids=trigger_tokens_tensor, 
                do_sample=True, 
                temperature=args.temperature, 
                max_length=max_length, 
                pad_token_id=tokenizer.pad_token_id,
            )
            after_time = time.time()
            if dev == "gpu":
                after_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            else:
                after_energy = get_cpu_reading()
            
            for x in out:
                if args.model != 'tloen/alpaca-lora-7b':
                    cnt_len = cal_real_length(x, tokenizer, max_length)
                else:
                    cnt_len = cal_real_length_alpaca(x, tokenizer, max_length)
                length_list.append(cnt_len)
                if cnt_len == max_length: sum_max += 1
                print(tokenizer.decode(x))
                print(cnt_len, '-----------')
            print(f"Part {cnt}, Time Cost: {time.time() - before_time}")
            time_delta   = after_time - before_time
            energy_delta = after_energy - before_energy
            point_list.append((cnt_len - input_length, time_delta, energy_delta))
        print(point_list)
    return trigger_info

from deepspeed.profiling.flops_profiler import FlopsProfiler

def eval_triggers_v3(trigger_list, device, args, model_existing=None):
    '''
        Not only running time, but also the power consumption is recorded.
    '''
    torch.cuda.empty_cache()
    transformers.deepspeed.unset_hf_deepspeed_config()

    if not isinstance(trigger_list[0], List):
        trigger_list = [trigger_list]


    batch_size = args.bs
    sample_time = args.sample_time
    max_length = args.max_length
    model_name = args.model
    if model_existing is None:
        tokenizer, model = get_model(model_name, args)
    else:
        tokenizer, model = model_existing
    
    prof = FlopsProfiler(model)
    
    trigger_info = []
    point_list = []
    for trigger_tokens in trigger_list:
        input_length = len(trigger_tokens)
        remaining_samples = sample_time
        cnt, sum_max = 0, 0
        
        length_list = []
        while remaining_samples > 0:

            bs = min(remaining_samples, batch_size)
            remaining_samples -= bs
            cnt += 1

            trigger_tokens_tensor = torch.tensor([trigger_tokens]).repeat(bs, 1).to(device)
            before_time = time.time()
            prof.start_profile()
            out = model.generate(
                input_ids=trigger_tokens_tensor, 
                do_sample=True, 
                temperature=args.temperature, 
                max_length=max_length, 
                pad_token_id=tokenizer.pad_token_id,
            )
            after_time = time.time()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            prof.end_profile()
            
            for x in out:
                if args.model != 'tloen/alpaca-lora-7b':
                    cnt_len = cal_real_length(x, tokenizer, max_length)
                else:
                    cnt_len = cal_real_length_alpaca(x, tokenizer, max_length)
                length_list.append(cnt_len)
                if cnt_len == max_length: sum_max += 1
                print(tokenizer.decode(x))
                print(cnt_len, '-----------')
            print(f"Part {cnt}, Time Cost: {time.time() - before_time}")
            time_delta   = after_time - before_time

            point_list.append((cnt_len, time_delta, flops))
        print(point_list)
    return trigger_info
