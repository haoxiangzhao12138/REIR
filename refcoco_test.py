import torch
import torch.nn.functional as F

model = torch.hub.load('/public/haoxiangzhao/weights/radio/NVlabs_RADIO_main', 'radio_model', version='/public/haoxiangzhao/weights/radio/radio_v2.5-h.pth.tar', adaptor_names='clip', source='local',)
# output = model(images)  # Inputs should have values between 0 and 1
# bb_summary, bb_features = output['backbone']
# clip_summary, clip_features = output['clip']  # These are the DFN CLIP embeddings

# To get the text embeddings
clip_adaptor = model.adaptors['clip']
tokens = clip_adaptor.tokenizer(['foo', 'bar'])
clip_text_embeddings = clip_adaptor.encode_text(tokens)

# B x B compatibility matrix from each image embedding to each text embedding (e.g. CLIP objective)
# alignment = F.normalize(clip_summary, dim=1) @ F.normalize(clip_text_embeddings.T, dim=0)