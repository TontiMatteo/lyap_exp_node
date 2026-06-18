from PIL import Image

gif = Image.open(r"C:\Users\m.tonti\Documents\lyap_exp\pics\circle_node\init_analysis_3exp\alpha_1.0\seed_0\hidden_flow.gif")
last_frame_index = gif.n_frames - 1

gif.seek(last_frame_index)
gif.convert("RGB").save(r"C:\Users\m.tonti\Documents\lyap_exp\pics\circle_node\init_analysis_3exp\alpha_1.0\seed_0\hidden_flow_ff.png")