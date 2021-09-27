def varitex_default_options():
    opt = {
        "dataset": "FFHQ",
        "image_h": 256,
        "image_w": 256,
        "latent_dim": 256,
        "texture_dim": 256,
        "texture_nc": 16,
        "nc_feature2image": 64,
        "feature2image_num_layers": 5,
        "nc_decoder": 32,
        "semantic_regions": list(range(1, 16))}
    return opt