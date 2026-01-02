import torch


def token_importance(model, input_ids, attention_mask):
    model.eval()
    input_ids = input_ids.clone().detach().requires_grad_(True)

    outputs = model.text_encoder.model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    score = cls_embedding.norm()
    score.backward()

    importance = input_ids.grad.abs().sum(dim=-1)
    return importance
