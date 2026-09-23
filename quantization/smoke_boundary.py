#!/usr/bin/env python3
"""Exercise Dots3 activation-boundary persistence without model weights."""

import tempfile
from pathlib import Path

import torch

from dots3_layer_boundary_store import (
    Dots3LayerBoundaryStore,
    LayerBoundaryError,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dots3-boundary-", dir="/work/state") as root_name:
        root = Path(root_name)
        journal = root / "errors.jsonl"
        journal.write_text("")
        store = Dots3LayerBoundaryStore(
            root / "boundaries",
            plan_sha256="0" * 64,
            family_join={"recipe": "smoke"},
            projection_checkpoint_root=root / "projections",
            error_journal_path=journal,
            hidden_size=4,
            activation_rank=3,
            routed_experts=2,
            first_target_layer=1,
            last_target_layer=1,
        )
        # This check isolates the activation and empty-replay-state contract;
        # production projection records are validated by their checkpoint store.
        entries = tuple(
            {"module": f"model.layers.1.mlp.experts.{expert}.{projection}"}
            for expert in range(2)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        store._validate_projection_entries = lambda _index, _entries: entries
        store._validate_completed_projection_index = (
            lambda _index, _entries: entries
        )
        outputs = [
            [torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4)],
            [torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)],
        ]
        kwargs = [{}, {}]
        masks = [None, None]
        manifest = store.commit(
            layer_index=1,
            layer_name="model.layers.1",
            layer_outputs=outputs,
            layer_input_kwargs=kwargs,
            position_ids=masks,
            attention_masks=masks,
            projection_entries=entries,
        )
        assert manifest["replay_state_shards"] == []
        restored = store.load_latest(
            layer_input_kwargs=kwargs,
            position_ids=masks,
            attention_masks=masks,
        )
        assert restored is not None and len(restored.layer_inputs) == 2
        for expected, actual in zip(outputs, restored.layer_inputs):
            torch.testing.assert_close(expected[0], actual[0])
        directory = next((root / "boundaries").iterdir())
        shard = directory / manifest["activation_shards"][0]["file"]
        with shard.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
        try:
            store.load_latest(
                layer_input_kwargs=kwargs,
                position_ids=masks,
                attention_masks=masks,
            )
        except LayerBoundaryError:
            pass
        else:
            raise AssertionError("corrupt activation shard was accepted")
    print("Dots3 activation boundary round-trip and corruption rejection passed")


if __name__ == "__main__":
    main()
