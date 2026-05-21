import torch
from diffusers import StableDiffusionPipeline
from token2attn.ptp_utils2 import prepare_bboxes, compute_cross_loss, compute_self_loss

def boxloss(pipeline, attention_store, latents, prompt, indices, fg_bounding_boxes, bg_bounding_boxes, device):
    
    topk_coef = 0.8
    dropout = 0.5
    margin = 0.1

    text_input = pipeline.tokenizer(prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        cond_text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0]

    #latens  传入

    bbox_masks = prepare_bboxes(fg_bounding_boxes, bg_bounding_boxes, 16, device)

    with torch.enable_grad():
        latents = latents.clone().detach().requires_grad_(True)
        pipeline.unet(latents, t, encoder_hidden_states=cond_text_embeddings).sample
        pipeline.unet.zero_grad()
        cross_attentions, self_attentions = attention_store.aggregate_attention(from_where=(['up']),)
        loss = compute_cross_loss(cross_attentions, indices, bbox_masks, topk_coef, dropout, margin) + compute_self_loss(self_attentions, indices, bbox_masks, topk_coef, dropout)

    return loss