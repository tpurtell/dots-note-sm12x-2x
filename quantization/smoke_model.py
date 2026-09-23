#!/usr/bin/env python3
"""Check the Dots3 GPTQModel adapter without loading checkpoint weights."""

import argparse

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.models.definitions.dots3_note import Dots3NoteQModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.source)
    assert config.model_type == "dots3_note", config.model_type
    Dots3NoteQModel.before_model_load(Dots3NoteQModel, args.source, False)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    assert len(model.model.layers) == 46
    assert len(model.model.layers[1].mlp.experts) == 256
    assert len(model.model.layers[45].mlp.experts) == 256
    assert model.model.layers[0].mlp.gate_proj.weight.device.type == "meta"
    for layer in (1, 2, 45):
        module = model.model.layers[layer].mlp.experts[255]
        assert set(dict(module.named_parameters())) == {
            "gate_proj.weight", "up_proj.weight", "down_proj.weight"
        }
    assert isinstance(Dots3NoteQModel.after_model_load(Dots3NoteQModel, model), type(model))
    print("Dots3 meta shell and 45 x 256 separate expert modules: OK")


if __name__ == "__main__":
    main()
