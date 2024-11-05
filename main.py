import torch

model_version = "radio_v2.5-l"  # for RADIOv2.5-L model (ViT-L/16)
#model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
#model_version="e-radio_v2" # for E-RADIO
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True,
                       skip_validation=True, adaptor_names='siglip')
model.cuda().eval()
x = torch.rand(1, 3, 512, 512, device='cuda')

if "e-radio" in model_version:
    model.model.set_optimal_window_size(x.shape[2:])  #where it expects a tuple of (height, width) of the input image.

# RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.
summary, spatial_features = model(x)

# RADIO also supports running in mixed precision:
with torch.autocast('cuda', dtype=torch.bfloat16):
    summary, spatial_features = model(x)

# If you'd rather pre-normalize the inputs, then you can do this:
conditioner = model.make_preprocessor_external()

# Now, the model won't change the inputs, and it's up to the user to call `cond_x = conditioner(x)` before
# calling `model(cond_x)`. You most likely would do this if you want to move the conditioning into your
# existing data processing pipeline.
with torch.autocast('cuda', dtype=torch.bfloat16):
    cond_x = conditioner(x)
    summary, spatial_features = model(cond_x)
