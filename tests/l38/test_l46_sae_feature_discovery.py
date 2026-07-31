import torch

from src.l38.l46_sae_feature_discovery import ESM_DIM, FEATURE_DIM, ReLUSAE


def test_relu_sae_architecture_matches_downloaded_checkpoint_shapes():
    # Real checkpoint state_dict keys/shapes confirmed via direct inspection
    # of huggingface.co/Elana/InterPLM-esm2-650m layer_24/ae_normalized.pt
    # before writing this class: bias [1280], encoder.weight [10240, 1280],
    # encoder.bias [10240], decoder.weight [1280, 10240], no decoder bias.
    sae = ReLUSAE(activation_dim=ESM_DIM, dict_size=FEATURE_DIM)
    state_dict = sae.state_dict()
    assert state_dict["bias"].shape == (ESM_DIM,)
    assert state_dict["encoder.weight"].shape == (FEATURE_DIM, ESM_DIM)
    assert state_dict["encoder.bias"].shape == (FEATURE_DIM,)
    assert state_dict["decoder.weight"].shape == (ESM_DIM, FEATURE_DIM)
    assert "decoder.bias" not in state_dict


def test_relu_sae_encode_produces_nonnegative_sparse_features():
    torch.manual_seed(0)
    sae = ReLUSAE(activation_dim=16, dict_size=64)
    x = torch.randn(5, 16)
    features = sae.encode(x)
    assert features.shape == (5, 64)
    assert (features >= 0).all()  # ReLU output


def test_relu_sae_decode_reconstructs_to_activation_dim():
    torch.manual_seed(0)
    sae = ReLUSAE(activation_dim=16, dict_size=64)
    features = torch.rand(5, 64)
    recon = sae.decode(features)
    assert recon.shape == (5, 16)


def test_relu_sae_forward_is_encode_then_decode():
    torch.manual_seed(0)
    sae = ReLUSAE(activation_dim=16, dict_size=64)
    x = torch.randn(3, 16)
    direct = sae.forward(x)
    composed = sae.decode(sae.encode(x))
    assert torch.allclose(direct, composed)
