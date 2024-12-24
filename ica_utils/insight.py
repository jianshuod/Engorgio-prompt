import torch
import torch.nn.functional as F

def present_theta_sample(token_ids, tokenizer):
    print('[Sampled from Theta]', tokenizer.decode(torch.tensor(token_ids), skip_special_tokens=True))

def represent_eos_probability(coeffs, tokenizer):
    slices = coeffs[0, :, tokenizer.eos_token_id]
    print('---------------------')
    print("Max:", slices.max(), slices.argmax())
    print("Mean:", slices.mean())
    print("Std:", slices.std())
    print('---------------------')
    return (slices.max().item(), slices.mean().item(), slices.std().item())

def check_trigger_sequence_loss(model, trigger_ids, embeddings, total_vocab_size, log_coeffs, device):
    inputs_embeds = embeddings[trigger_ids].detach().unsqueeze(0)
    pred = model(inputs_embeds=inputs_embeds).logits
    pred_t = pred.contiguous().view(-1, total_vocab_size)[:-1]
    target_t = F.softmax(log_coeffs[1:])
    loss2 = F.cross_entropy(pred_t, torch.tensor(trigger_ids[1:], device=device))
    print(loss2.item())
    return loss2