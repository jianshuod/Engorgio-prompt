import torch
import torch.nn.functional as F

from .prepare import TemplateFactory

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    cel_list = torch.mul(log_likelihood, target)
    if reduction == 'average':
        loss = torch.sum(cel_list) / batch
    else:
        loss = torch.sum(cel_list)
    return loss

def SelfMentorLoss(pred_t, target_t, args):
    loss1 = SoftCrossEntropy(pred_t, target_t, reduction='sum')
    return loss1

def EOSProbLoss(pred_t, args):
    normalized_pred_t = F.softmax(pred_t, dim=1)
    loss2 = torch.sum(normalized_pred_t.view(-1, args.total_vocab_size)[:, args.eos_token_id])
    return args.alpha * loss2

def total_loss_v1(pred_t, target_t, template_fac:TemplateFactory, args, need_detail=False, loss_options=[1, 2]):
    '''
        pred_t    \in   (max_length, V)
        target_t  \in   (max_length - template_length, V)
    '''
    prefix_length = template_fac.prefix_length
    trigger_seq_length = template_fac.trigger_token_length
    response_length = template_fac.response_offset
    output_length = template_fac.template_w_trigger_length
    theta_length = args.max_length - template_fac.template_length

    # Self-Mentor Loss
    # Part 1 -- Trigger Part [prefix_length + 1, response_offset)
    loss1_1 = SelfMentorLoss(pred_t[prefix_length:response_length - 1], target_t[1:trigger_seq_length], args)
    # Part 2 -- Output Part
    loss1_2 = SelfMentorLoss(pred_t[output_length - 1:-1], target_t[trigger_seq_length:], args)
    loss1 = (loss1_1 + loss1_2) / theta_length

    if args.esc_loss_version == 0:
        loss2 = EOSProbLoss(pred_t[prefix_length:], args) # v0
    else:
        loss2 = EOSProbLoss(pred_t[output_length:], args) # v1
    loss = 0
    if 1 in loss_options: loss += loss1
    if 2 in loss_options: loss += loss2

    return  loss, (loss1.item(), loss2.item()/args.alpha) if need_detail else ()

def total_loss_v0(pred_t, target_t, args, need_detail=False, loss_options=[1, 2]):
    theta_length = args.theta_length
    loss1 = SelfMentorLoss(pred_t, target_t, args) / theta_length
    loss2 = EOSProbLoss(pred_t, args)
    loss = 0
    if 1 in loss_options: loss += loss1
    if 2 in loss_options: loss += loss2

    return  loss, (loss1.item(), loss2.item()) if need_detail else ()

# def diversity_loss(args):
#     if args.contrastive_loss:
#         hidden_state = hidden_state / hidden_state.norm(dim=1, keepdim=True)
#         con_similarity = hidden_state @ hidden_state.t()
#         loss_contrastive = torch.max(torch.tril(con_similarity, diagonal=-1), dim=1).values.sum()
#         loss = loss + 0.005*loss_contrastive
#     else:
#         with torch.no_grad():
#             hidden_state = hidden_state / hidden_state.norm(dim=1, keepdim=True)
#             con_similarity = hidden_state @ hidden_state.t()
#             loss_contrastive = torch.max(torch.tril(con_similarity, diagonal=-1), dim=1).values.sum()