import os
import time
import random
import argparse

import torch
import torch.nn.functional as F

# import config
from ica_utils.model import get_model, load_deepspeed
from ica_utils.loss import *
from ica_utils.eval import eval_triggers, eval_triggers_v2, eval_triggers_v3
from ica_utils.util import set_seed
from ica_utils.prepare import TemplateFactory, get_normal_init
from ica_utils.ctl import Converge_check

def main(args):

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer, model = get_model(args.model, args)
    if args.deepspeed: dschf, model = load_deepspeed(model, args)
    try:
        total_vocab_size = model.get_output_embeddings().out_features
    except:
        total_vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab().keys())
    print(dir(model))

    args.total_vocab_size = total_vocab_size
    args.eos_token_id = model.config.eos_token_id
    print(args.eos_token_id)
    embeddings = model.get_input_embeddings()(torch.arange(0, total_vocab_size).long().to(device)).detach()
        # prefix_sentence = "StableLM is a helpful and harmless open-source AI language model developed"
    trigger_seq_length = args.trigger_token_length

    global_triggers = []; losse1s = []; losse2s = []
    # -----------------[Init the Env]------------------ 
    template_fac = TemplateFactory(
        args.model, trigger_seq_length, tokenizer, embeddings
    )
        # template_fac.add_additional_prompt(prefix_sentence)
    template_len = template_fac.template_length
    theta_length = args.max_length - template_len
    args.theta_length = theta_length
    checker = Converge_check()

    # -----------------[Init the Trigger Theta]------------------ 
    log_coeffs = torch.zeros(theta_length, total_vocab_size, dtype=embeddings.dtype)
    if args.normal_init:
        normal_init = get_normal_init(tokenizer)
        init_length = len(normal_init)
        trigger_ids = normal_init + [random.randint(0, total_vocab_size) for i in range(theta_length - init_length)]
    else:
        trigger_ids = [random.randint(0, total_vocab_size - 1) for _ in range(theta_length)]
    indices = torch.arange(len(trigger_ids)).long()
    log_coeffs[indices, torch.LongTensor(trigger_ids)] = args.initial_coeff
    log_coeffs = log_coeffs.to(device)
    log_coeffs.requires_grad = True
    trigger_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).tolist()


    # -----------------[Training]------------------ 
    optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
    loss_options = args.loss_opt
    s_time = time.time()
    for i in range(args.num_iters):
        # Warmup using <EOS> Escape Loss
        if i == args.esc_loss_warmup_iters and args.warmup_filter:
            log_top1 = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).tolist()
            with torch.no_grad():
                log_coeffs.zero_()
                log_coeffs[indices, torch.LongTensor(log_top1)] = args.warmup_initial_coeff
        if i < args.esc_loss_warmup_iters: 
            args.alpha = args.warmup_alpha
            loss_options = [2]
            for param_group in optimizer.param_groups: param_group["lr"] = args.warmup_lr
        else: 
            args.alpha = args.opt_alpha
            loss_options = args.loss_opt
            for param_group in optimizer.param_groups: param_group["lr"] = args.lr

        if args.alternate:
            if i % 2 == 0: loss_options = [2]
            else: loss_options = [1]

        if args.trigger_esc_eos:
            with torch.no_grad():
                log_coeffs[:trigger_seq_length, args.eos_token_id] = torch.finfo(log_coeffs.dtype).min
        optimizer.zero_grad()

        coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0), hard=False)

        # if args.load_in_8bit: coeffs = coeffs.half()
        inputs_embeds = (coeffs @ embeddings)
        inputs_embeds_x = template_fac.get_input_embeddings(inputs_embeds)

        pred = model(inputs_embeds=inputs_embeds_x).logits
        pred_t = pred.contiguous().view(-1, total_vocab_size)
        target_t = F.softmax(log_coeffs, dim=1)

        loss, (loss1, loss2) = total_loss_v1(pred_t, target_t, template_fac, args, need_detail=True, loss_options=loss_options)
        losse1s.append([(i + 1), loss1]); losse2s.append([(i + 1), loss2])
        loss.backward()

        optimizer.step()

        trigger_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).tolist()[:trigger_seq_length]
        if checker(trigger_ids): break
        if (i + 1) % args.log_interval == 0: global_triggers.append(template_fac.get_input_tokens(checker.new_state))
        print(f'[Epoch {i}]({checker.counter}), loss:{loss.item()}, CEL:{loss1}, WOL:{loss2}')
        if i == 100: print(f"Time Cost: {time.time() - s_time}")
        
    global_triggers.append(template_fac.get_input_tokens(checker.new_state))
    # if args.save_path != '': torch.save(global_triggers, os.path.join(config.save_dir, f"{args.save_path}.pth"))

    # -----------------[Evaluation]------------------ 
    # del tokenizer, model
    model.gradient_checkpointing_disable()
    if args.deepspeed: del dschf
    print(losse1s); print(losse2s); 
    # print(losse1s); print(losse2s); print(global_triggers)
    if args.eval_v2:
        print(eval_triggers_v3(global_triggers, device, args, (tokenizer, model))); 
    else:
        print(eval_triggers(global_triggers, device, args, (tokenizer, model))); 
    print(global_triggers)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # [Basic], Experiment Settings
    parser.add_argument("--model", default='', type=str)
    parser.add_argument("--save_path", default='', type=str)
    parser.add_argument("--seed", default=313406, type=int, help="Trial Seed")
    parser.add_argument("--log_interval", default=500, type=int, help="Every x iters, eval the theta")

    # [Warmup]
    parser.add_argument("--warmup_lr", default=0.1, type=float, help="warmup learning rate")
    parser.add_argument("--esc_loss_warmup_iters", default=0, type=int)
    parser.add_argument("--warmup_alpha", default=1, type=float, help="weight of the wiping out loss")
    parser.add_argument("--warmup_filter", action="store_true")
    parser.add_argument("--warmup_initial_coeff", default=5, type=int, help="initial log coefficients")

    # [Training], Design Settings
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--num_iters", default=5000, type=int, help="number of epochs to train for")
    parser.add_argument("--opt_alpha", default=1, type=float, help="weight of the wiping out loss")
    parser.add_argument("--loss_opt", type=int, nargs='+')
    parser.add_argument("--esc_loss_version", default=0, type=int)
    parser.add_argument("--sm_loss_version", default=0, type=int)
    parser.add_argument("--trigger_esc_eos", action="store_true")
    parser.add_argument("--alternate", action="store_true")


    # [Initialization], Theta Settings
    parser.add_argument("--trigger_token_length", default=32, type=int, help='how many subword pieces in the trigger')
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--initial_coeff", default=0, type=int, help="initial log coefficients")
    parser.add_argument("--normal_init", action="store_true")

    # [Inference], Evaluation Settings
    parser.add_argument("--bs", "--batch_size", default=1, type=int, help="[Inference], batch size for inference")
    parser.add_argument("--sample_time", default=200, type=int, help="[Inference], total sample time to calculate avg_rate")
    parser.add_argument("--temperature", default=0.7)
    parser.add_argument("--top_k", default=0, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--eval_v2", action="store_true")


    # [DeepSpeed], Acceleration Settings
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device_id", default=0, type=int, help="device id")

    args = parser.parse_args()

    print(args)
    main(args)