#!/usr/bin/env python3
"""CPU-only timeout continuation integrity checks; no network or GPUs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import continue_spark_timeout as c


class TimeoutTests(unittest.TestCase):
    def test_only_timeout_ast_changes(self):
        original=ast.parse(c.WORKLOAD.read_text());effective=ast.parse(c.transformed())
        changes=0
        for node in ast.walk(original):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='urlopen':
                for kw in node.keywords:
                    if kw.arg=='timeout':
                        self.assertEqual(kw.value.value,900);kw.value.value=3600;changes+=1
        self.assertEqual(changes,1)
        self.assertEqual(ast.dump(original),ast.dump(effective))

    def fixture(self,d):
        p=Path(d);(p/'prior.json').write_text('{"complete":true}')
        m={'identity':{'rhea':{'image_id':'fixed'},'moa':{'image_id':'fixed'}},'source_sha256':{},'plan':[{'name':'seven'},{'name':'retrieval-522144'}]}
        ledger={'schema':'dots3-spark-timeout-continuation-v1','identity':copy.deepcopy(m['identity']),
                'timeout_seconds':{'original':900,'continued':3600},
                'effective_workloads_sha256':hashlib.sha256(c.transformed().encode()).hexdigest(),
                'sources':{},'preserved_files':{'prior.json':c.digest(p/'prior.json')},
                'remaining_stages':['retrieval-522144'],'preserved_stages':['seven']}
        (p/c.LEDGER).write_text(json.dumps(ledger));return p,m,ledger

    def test_valid_and_reject_receipt_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,l=self.fixture(d);c.verify(p,m)
            (p/'prior.json').write_text('{}')
            with self.assertRaisesRegex(AssertionError,'prior evidence changed'):c.verify(p,m)

    def test_reject_identity_timeout_and_sources(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,l=self.fixture(d)
            bad=copy.deepcopy(m);bad['identity']['rhea']['image_id']='wrong'
            with self.assertRaises(AssertionError):c.verify(p,bad)
            l['timeout_seconds']['continued']=7200;(p/c.LEDGER).write_text(json.dumps(l))
            with self.assertRaises(AssertionError):c.verify(p,m)
            l['timeout_seconds']['continued']=3600;l['effective_workloads_sha256']='wrong';(p/c.LEDGER).write_text(json.dumps(l))
            with self.assertRaises(AssertionError):c.verify(p,m)


if __name__=='__main__':unittest.main()
