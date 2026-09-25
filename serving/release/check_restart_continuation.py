#!/usr/bin/env python3
"""CPU fixtures for strict restart provenance and preservation of completed work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import restart_continuation as audit


class Validator:
    @staticmethod
    def validate(step, artifact, limit):
        assert json.loads(artifact.read_text()) == {'complete': True}
        assert limit == 524288


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'run';self.out.mkdir()
        self.old={'id':'same-container','image_id':'sha256:original','started_at':'before',
                  'restart_count':0,'args':['model/snapshots/REV','--max-model-len','524288'],
                  'environment_sha256':'same-env'}
        self.new=self.old|{'started_at':'after','restart_count':1}
        for name in (*audit.TOOLS,'serving/release/qualify_rtx.py','serving/benchmarks/context.py'):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('unchanged-source\n')
        self.manifest={'schema':'dots3-rtx-release-qualification-v2','identity':self.old,
                       'source_sha256':{p:audit.sha(self.root/p) for p in ['serving/release/qualify_rtx.py','serving/benchmarks/context.py']},
                       'limits':{'max_model_len':524288},
                       'plan':[{'name':f'finished-{i}'} for i in range(14)]+[{'name':x} for x in ['context-524032','retrieval-8192','retrieval-522144']]}
        self.write(self.out/'manifest.json',self.manifest)
        for step in self.manifest['plan'][:14]:
            d=self.out/step['name'];d.mkdir();self.write(d/'result.json',{'complete':True})
            self.write(d/'receipt.json',{'identity':self.old,'artifact':'result.json','sha256':audit.sha(d/'result.json'),'validated':{'complete':True}})
        self.evidence=self.root/'failure.json';self.write(self.evidence,{'xid':79})

    def write(self,p,data):p.write_text(json.dumps(data))

    def prepare(self):
        return audit.prepare(self.out,self.manifest,self.new,Validator,'User authorized continuation after reboot',self.evidence,self.root)

    def ledger(self):
        a=self.prepare();self.write(self.out/audit.LEDGER,a);return a

    def test_preserves_fourteen_and_runs_only_three(self):
        before={str(p):p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        a=self.ledger();audit.verify(self.out,self.manifest,self.new,self.root)
        self.assertEqual(len(a['preserved_stages']),14)
        self.assertEqual(a['remaining_stages'],['context-524032','retrieval-8192','retrieval-522144'])
        for p,raw in before.items():self.assertEqual(Path(p).read_bytes(),raw)
        for s in a['preserved_stages']:self.assertEqual(audit.stage_identity(a,s),self.old)
        for s in a['remaining_stages']:self.assertEqual(audit.stage_identity(a,s),self.new)

    def test_reject_wrong_image_args_environment(self):
        for key,value in [('image_id','different'),('args',['different-model']),('environment_sha256','different')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                audit.same_profile(self.old,self.new|{key:value})

    def test_reject_changed_workload_source(self):
        (self.root/'serving/benchmarks/context.py').write_text('changed')
        with self.assertRaisesRegex(AssertionError,'source changed'):self.prepare()

    def test_reject_changed_artifact_even_before_ledger(self):
        (self.out/'finished-0/result.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError,'artifact changed'):self.prepare()

    def test_reject_rewritten_receipt_and_artifact_after_ledger(self):
        self.ledger();p=self.out/'finished-0/result.json';p.write_text('{"complete": true}\n')
        r=audit.read(p.parent/'receipt.json');r['sha256']=audit.sha(p);self.write(p.parent/'receipt.json',r)
        with self.assertRaisesRegex(AssertionError,'preserved evidence changed'):
            audit.verify(self.out,self.manifest,self.new,self.root)

    def test_reject_altered_prior_identity(self):
        p=self.out/'finished-0/receipt.json';r=audit.read(p);r['identity']=self.new;self.write(p,r)
        with self.assertRaisesRegex(AssertionError,'prior stage identity'):self.prepare()

    def test_reject_manifest_failure_or_tool_source_mutations(self):
        self.ledger()
        for p in [self.out/'manifest.json',self.evidence,self.root/audit.TOOLS[0]]:
            original=p.read_bytes();p.write_bytes(original+b'\n')
            with self.subTest(path=p),self.assertRaises(AssertionError):audit.verify(self.out,self.manifest,self.new,self.root)
            p.write_bytes(original)

    def test_reject_unrecorded_second_restart(self):
        self.ledger()
        with self.assertRaisesRegex(AssertionError,'another restart'):
            audit.verify(self.out,self.manifest,self.new|{'started_at':'third'},self.root)

    def test_reject_path_escape(self):
        with self.assertRaisesRegex(AssertionError,'escapes'):audit.contained(self.out,'../secret')


if __name__=='__main__':unittest.main()
