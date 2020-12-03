import torch

# x2_path = '../pretrained/LAPAR_A_x2_200K.pth'
x2_path = '../pretrained/LAPAR_A_x4_200K.pth'
x3_path = '../pretrained/LAPAR_A_x3.pth'

x2_state = torch.load(x2_path, map_location=torch.device('cpu'))
x3_state = torch.load(x3_path, map_location=torch.device('cpu'))

new_dict = dict()
new_path = '../pretrained/LAPAR_A_x4_200K_new.pth'
for key, val in x2_state.items():
    if key != 'K':
        new_dict[key] = val
for key, val in x3_state.items():
    if key == 'w_conv.rgb_mean' or key == 'decom_conv.weight':
        new_dict[key] = val
torch.save(new_dict, new_path)
