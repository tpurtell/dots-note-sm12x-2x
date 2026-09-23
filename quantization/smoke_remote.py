#!/usr/bin/env python3
"""Verify the coordinator can authenticate the live Spark worker."""

import json
import sys

from gptqmodel.utils.exl3_remote import remote_client_from_provenance


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as stream:
        provenance = json.load(stream)
    client = remote_client_from_provenance(provenance)
    if client is None or len(client.endpoints) != 1:
        raise RuntimeError("expected one qualified remote worker")
    client.qualify(client.endpoints[0])
    print("rhea -> moa EXL3 worker authenticated and identity matched")


if __name__ == "__main__":
    main()
