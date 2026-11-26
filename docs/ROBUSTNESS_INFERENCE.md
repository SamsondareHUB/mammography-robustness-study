# Robustness inference infrastructure

The CPU-only inference matrix supports the two baseline architectures, training seeds 42, 43 and 44, and the 13 conditions (clean plus 12 degradations). Transformations are applied to the resized content rectangle, before ImageNet normalization; padding remains black. Each case/condition uses the same input tensor across the six checkpoints. The original clean inference path is retained and checked against integrated inference in the same runtime.

Run the regression suite and the synthetic integration check from the repository root:

```bash
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
python -m scripts.check_robustness_integration
```

The integration check uses synthetic images and randomly initialized model weights. It checks all 78 model/seed/condition combinations, shared inputs, clean path equivalence, coverage and metadata exports. No private dataset, trained checkpoint, or real prediction is included.

The inference matrix is a library component, not a full private-data execution or provenance runner. Exact agreement with archived real clean predictions remains a separate prerequisite; same-runtime synthetic equality is not evidence of historical bitwise reproducibility. Do not run the full test degradation experiment until the clean-reference procedure is approved and the private-run preflight and artifact recording are implemented. The paired statistical evaluator is outside this change.
