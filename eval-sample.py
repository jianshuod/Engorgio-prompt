import os
import time
import random
import argparse

import torch
import torch.nn.functional as F

import numpy as np

import yaml

# import config
from ica_utils.model import get_model, load_deepspeed
from ica_utils.loss import *
from ica_utils.eval import eval_triggers, cal_real_length, cal_real_length_alpaca
from ica_utils.util import set_seed
from ica_utils.prepare import TemplateFactory, load_data


def main(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer, model = get_model(args.model, args)
    if args.deepspeed: dschf, model = load_deepspeed(model, args)
    # total_vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab().keys())
    total_vocab_size = model.get_output_embeddings().out_features
    args.total_vocab_size = total_vocab_size
    args.eos_token_id = model.config.eos_token_id
    embeddings = model.get_input_embeddings()(torch.arange(0, total_vocab_size).long().to(device)).detach()
    trigger_seq_length = args.trigger_token_length

    # -----------------[Init the Env]------------------ 
    template_fac = TemplateFactory(
        args.model, trigger_seq_length, tokenizer, embeddings
    )

    model.eval()

    specific_prompt = "Today is a sunny day. Let us talk about the weather endlessly."

    normal_samples = [specific_prompt]

    print(f"\n")
    sum_max = 0
    length_list = []
    input_length_list = []
    raw_input_length_list = []
    sample_time = len(normal_samples)
    max_length  = args.max_length
    batch_size = args.bs
    s_time = time.time()
    start = 0 if 'stable' in args.model else 1
    for sample in normal_samples:
        print(f"Sample {sample}")
        inputs = tokenizer(sample)['input_ids']
        end = len(inputs)
        if args.cut:
            end = min(end, start+32)
        inputs = inputs[start:end]
        print(len(inputs))
        print(tokenizer.decode(inputs))
        raw_input_length_list.append(len(inputs))
        inputs = template_fac.get_input_tokens(inputs)
        print(tokenizer.decode(inputs))
        
        remaining_samples = args.sample_time
        cnt= 0
        
        while remaining_samples > 0:
            bs = min(remaining_samples, batch_size)
            remaining_samples -= bs
            cnt += 1

            trigger_tokens_tensor = torch.tensor([inputs]).repeat(bs, 1).to(device)
        
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
                print(tokenizer.decode(x, skip_special_tokens=False))
                print(len(inputs), cnt_len, '-----------')
    print(np.mean(input_length_list))
    print(np.mean(raw_input_length_list))
    sum_time = time.time() - s_time
    avg_time = sum_time / sample_time
    avg_len = np.mean(length_list)
    std_len = np.std(length_list)
    avg_rate = sum_max / sample_time
    ratio = np.sum(length_list) / np.sum(input_length_list) - 1
    print(avg_time, avg_len, std_len, avg_rate, ratio)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # [Basic], Experiment Settings
    parser.add_argument("--model", default='', type=str)
    parser.add_argument("--save_path", default='test.txt', type=str)
    parser.add_argument("--output_file", default='test.txt', type=str)
    parser.add_argument("--seed", default=0, type=int, help="Trial Seed")
    parser.add_argument("--log_interval", default=500, type=int, help="Every x iters, eval the theta")

    # [Training], Design Settings
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--num_iters", default=5000, type=int, help="number of epochs to train for")
    parser.add_argument("--alpha", default=1, type=float, help="weight of the wiping out loss")
    parser.add_argument("--loss_opt", type=int, nargs='+')
    parser.add_argument("--esc_loss_version", default=0, type=int)
    parser.add_argument("--trigger_esc_eos", action="store_true")
    parser.add_argument("--warmup_lr", default=0.1, type=float, help="warmup learning rate")
    parser.add_argument("--esc_loss_warmup_iters", default=0, type=int)
    parser.add_argument("--warmup_filter", action="store_true")
    parser.add_argument("--warmup_initial_coeff", default=5, type=int, help="initial log coefficients")


    # [Initialization], Theta Settings
    parser.add_argument("--trigger_token_length", default=32, type=int, help='how many subword pieces in the trigger')
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--initial_coeff", default=5, type=int, help="initial log coefficients")
    parser.add_argument("--normal_init", action="store_true")

    # [Inference], Evaluation Settings
    parser.add_argument("--bs", "--batch_size", default=1, type=int, help="[Inference], batch size for inference")
    parser.add_argument("--sample_time", default=200, type=int, help="[Inference], total sample time to calculate avg_rate")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_k", default=0, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)

    # [DeepSpeed], Acceleration Settings
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device_id", default=0, type=int, help="device id")


    parser.add_argument("--sample_num", default=100, type=int)
    parser.add_argument("--cut", action="store_true")
    parser.add_argument("--shareGPT_only", action="store_true")
    parser.add_argument("--alpaca_only", action="store_true")

    args = parser.parse_args()

    print(args)
    main(args)