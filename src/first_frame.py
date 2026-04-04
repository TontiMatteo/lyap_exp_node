from PIL import Image

gif = Image.open(r"C:\Users\m.tonti\Documents\lyap_exp\pics\circle_node\xavier_normal\linear_aug_T=1\baseg_tanh_hidden_flow_45.gif")
gif.seek(0)
gif.convert("RGB").save(r"C:\Users\m.tonti\Documents\lyap_exp\pics\circle_node\xavier_normal\linear_aug_T=1\baseg_tanh_hidden_flow_45_frame0.png")