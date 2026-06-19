import torch
from src.experiments.run_synthetic import (
    generate_unconditional,
    generate_guided_single,
    generate_composed_and,
    generate_composed_not,
    evaluate_boolean_accuracy,
)
from src.models.ctmc_flow import CTMCDenoiser
from src.models.prob_path_flow import ProbPathDenoiser
from src.models.classifier import TimeConditionalClassifier
from src.data.synthetic import property_starts_with_0, property_contains_pattern


def test_generate_unconditional_ctmc():
    model = CTMCDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_unconditional(model, n=10, K=4, L=8, num_steps=10, mode="ctmc")
    assert samples.shape == (10, 8)
    assert samples.min() >= 0 and samples.max() < 4


def test_generate_unconditional_prob_path():
    model = ProbPathDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_unconditional(model, n=10, K=4, L=8, num_steps=10, mode="prob_path")
    assert samples.shape == (10, 8)


def test_generate_guided_single():
    model = ProbPathDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    clf = TimeConditionalClassifier(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_guided_single(
        model, clf, n=10, K=4, L=8, num_steps=10, gamma=1.0, mode="prob_path"
    )
    assert samples.shape == (10, 8)


def test_generate_composed_and():
    model = ProbPathDenoiser(K=4, L=8, hidden_dim=32, num_layers=1)
    clf_a = TimeConditionalClassifier(K=4, L=8, hidden_dim=32, num_layers=1)
    clf_b = TimeConditionalClassifier(K=4, L=8, hidden_dim=32, num_layers=1)
    samples = generate_composed_and(
        model, [clf_a, clf_b], n=10, K=4, L=8, num_steps=10,
        gammas=[1.0, 1.0], mode="prob_path"
    )
    assert samples.shape == (10, 8)


def test_evaluate_boolean_accuracy():
    seqs = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3],
                         [1, 2, 3, 0, 1, 2, 3, 0]])
    prop_fns = [property_starts_with_0]
    acc = evaluate_boolean_accuracy(seqs, prop_fns, mode="and")
    assert 0.0 <= acc <= 1.0
    assert acc == 0.5  # only first seq starts with 0
