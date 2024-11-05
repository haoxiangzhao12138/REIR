import torch

# model = torch.load('/public/haoxiangzhao/weights/uninext/model_final.pth')
model = torch.load('/public/haoxiangzhao/weights/uninext/uninext_VIT-H.pth')

print(model.keys())

# 定义旧的参数名和新的参数名
old_ffn_params = [
    'ffn.layers.0.0.weight', 'ffn.layers.0.0.bias',
    'ffn.layers.1.weight', 'ffn.layers.1.bias',
    'norms.0.weight', 'norms.0.bias',
    'norms.1.weight', 'norms.1.bias',
    'norms.2.weight', 'norms.2.bias'

]

new_ffn_params = [
    'linear1.weight', 'linear1.bias',
    'linear2.weight', 'linear2.bias',
    'norm1.weight', 'norm1.bias',
    'norm2.weight', 'norm2.bias',
    'norm3.weight', 'norm3.bias'
]


# 遍历字典并替换键
def replace_keys_in_dict(input_dict, old_keys, new_keys):
    if len(old_keys) != len(new_keys):
        raise ValueError("Old and new key lists must have the same length.")
    
    updated_dict = {}
    
    for key, value in input_dict.items():
        replaced = False
        for old_key, new_key in zip(old_keys, new_keys):
            if key.endswith(old_key):  # 检查key是否以old_key结尾
                updated_dict[new_key] = value  # 用new_key替换
                replaced = True
                break
        if not replaced:
            if 'text_encoder' in key:
                continue
            if 'backbone' in key:
                continue
            updated_dict[key] = value  # 没有匹配到则保留原有键值对
    
    return updated_dict

# 执行替换
updated_dict = replace_keys_in_dict(model, old_ffn_params, new_ffn_params)

# 打印结果
print("原字典:", model.keys())
print("更新后的字典:", updated_dict.keys())

radio = torch.hub.load('/home/haoxiangzhao/.cache/torch/hub/NVlabs_RADIO_main', 'radio_model', checkpoint_path='/home/haoxiangzhao/.cache/torch/hub/checkpoints/radio-v2.5-l_half.pth.tar', version='radio_v2.5-l', progress=True,
                                    skip_validation=True, source='local', )
print(radio.state_dict().keys())

radio_dict = {}
for key, value in radio.state_dict().items():
    name = 'detr.backbone.radio.' + key
    radio_dict[name] = value
updated_dict.update(radio_dict)
# TODO 改backbone的参数名字

torch.save(updated_dict, '/public/haoxiangzhao/weights/uninext/fixed_uninext_VIT-H.pth')
pass
# Let's write a Python function that takes the FFN parameters from the first model naming style
# and transforms them to the second model's naming style.

# We'll iterate through all layers and make the necessary substitutions to change
# the parameter naming convention from the first model style to the second model style.

def convert_ffn_params_to_new_format(old_params):
    new_params = []
    for param in old_params:
        # Replace 'encoder.layers.X' to 'detr.detr.transformer.encoder.layers.X'
        param = param.replace("encoder.layers.", "detr.detr.transformer.encoder.layers.")
        
        # Replace the 'ffn.layers' pattern into 'linear1', 'linear2' and norms pattern
        if 'ffn.layers.0.0' in param:
            param = param.replace("ffn.layers.0.0", "linear1")
        elif 'ffn.layers.1' in param:
            param = param.replace("ffn.layers.1", "linear2")
        elif 'norms.0' in param:
            param = param.replace("norms.0", "norm1")
        elif 'norms.1' in param:
            param = param.replace("norms.1", "norm2")
        
        # Add the transformed parameter name to the new list
        new_params.append(param)
    
    return new_params

# Test the function on a sample input
old_ffn_params = [
    'encoder.layers.0.ffn.layers.0.0.weight', 'encoder.layers.0.ffn.layers.0.0.bias',
    'encoder.layers.0.ffn.layers.1.weight', 'encoder.layers.0.ffn.layers.1.bias',
    'encoder.layers.0.norms.0.weight', 'encoder.layers.0.norms.0.bias',
    'encoder.layers.0.norms.1.weight', 'encoder.layers.0.norms.1.bias'
]

new_ffn_params = convert_ffn_params_to_new_format(old_ffn_params)
new_ffn_params