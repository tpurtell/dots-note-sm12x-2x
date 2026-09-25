#!/usr/bin/env python3
"""CPU-only evidence inheritance tests: no HTTP, SSH, Docker or GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import evidence_lineage as lineage
import qualify_spark
import report


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source={h:{'image_id':'sha256:old-'+h,'args':['serve','/cache/snapshots/REV','--max-model-len','524288'],
                        'started_at':'old','selected_environment':['GC_POLICY=old']} for h in ('rhea','moa')}
        runtimes={}
        for h,r in self.source.items():
            p=self.root/(h+'.json');p.write_text(json.dumps(r));runtimes[h]=lineage.descriptor(p,self.root)
        p=self.root/'prefill.json'
        p.write_text(json.dumps({'schema':'dots3-prefill-depth.v1','runs_per_point':3,'points':[{'prompt_tokens':8192,'runs':[{
            'prompt_sha256':str(i),'usage':{'prompt_tokens':8192,'completion_tokens':1},'cached_tokens':None,'ttft_seconds':10+i}for i in range(3)]}]}))
        self.entry={'source_kind':'prefill','artifact':lineage.descriptor(p,self.root),'source_runtimes':runtimes,
                    'sample_count':3,'output_tokens_per_request':1,'depth':8192,
                    'coverage':'Prior image/profile, one-token prefill only; no decode proof','supporting_evidence':[]}
        self.spec={'schema':'dots3-prior-evidence-v1','authorization':'preserve-completed-tests-with-explicit-prior-profile-provenance','stages':{'context-8192':self.entry}}
        self.path=self.root/'spec.json';self.path.write_text(json.dumps(self.spec))
        self.target={h:r|{'image_id':'sha256:final','started_at':'new','selected_environment':['GC_POLICY=new']}for h,r in self.source.items()}
        self.identity={h:{k:r[k]for k in ('image_id','args','started_at')}for h,r in self.target.items()}
        self.step=lineage.apply_plan([{'name':'context-8192','requests':4,'format':'jsonl'}],self.path,self.root)[0]

    def materialize(self):
        d=self.root/'out';d.mkdir()
        return lineage.materialize(self.step,d,self.identity,self.target,self.root)

    def test_no_requests_and_null_decode(self):
        self.assertEqual(self.step['requests'],0)
        p=self.materialize();lineage.validate_inherited(self.step,p,self.identity,self.root,self.target)
        result=lineage.metrics(self.step,p,report.stage_metrics)
        self.assertIsNone(result['points'][0]['decode_tokens_per_second_median'])
        self.assertEqual(result['points'][0]['completion_tokens'],[1])
        self.assertEqual(result['points'][0]['samples'],3)
        self.assertEqual(result['evidence_provenance']['mode'],'inherited')
        self.assertIn('GC_POLICY',result['evidence_provenance']['profile_differences']['rhea']['environment'])

    def test_reject_changed_source_and_wrong_budget(self):
        e=copy.deepcopy(self.entry);e['output_tokens_per_request']=256
        with self.assertRaisesRegex(AssertionError,'output budget'):lineage.validate_source(e,self.root)
        (self.root/'prefill.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError,'source archive changed'):lineage.validate_source(self.entry,self.root)

    def test_reject_changed_checkpoint(self):
        self.target['rhea']['args']=['serve','/cache/snapshots/DIFFERENT'];self.identity['rhea']['args']=self.target['rhea']['args']
        with self.assertRaisesRegex(AssertionError,'checkpoint changed'):self.materialize()

    def test_reject_false_final_identity_or_environment(self):
        p=self.materialize();bad=copy.deepcopy(self.identity);bad['rhea']['image_id']='wrong'
        with self.assertRaisesRegex(AssertionError,'target runtime changed'):lineage.validate_inherited(self.step,p,bad,self.root)
        bad_runtime=copy.deepcopy(self.target);bad_runtime['rhea']['selected_environment']=['GC_POLICY=wrong']
        with self.assertRaisesRegex(AssertionError,'final runtime profile differs'):lineage.validate_inherited(self.step,p,self.identity,self.root,bad_runtime)

    def test_reject_changed_archive_and_fake_sample_count(self):
        p=self.materialize();raw=p.parent/'source-000.raw';raw.write_bytes(raw.read_bytes()+b' ')
        with self.assertRaises(AssertionError):lineage.validate_inherited(self.step,p,self.identity,self.root)
        e=copy.deepcopy(self.entry);e['sample_count']=4
        with self.assertRaisesRegex(AssertionError,'sample count'):lineage.validate_source(e,self.root)

    def test_original_spark_plan_unchanged_without_optin(self):
        args=SimpleNamespace(base_url='http://unused',model='unused',max_model_len=524288)
        steps=qualify_spark.plan(args)
        self.assertTrue(all(s.get('evidence_mode')!='inherited'for s in steps))
        self.assertEqual(next(s['requests']for s in steps if s['name']=='context-8192'),4)


if __name__=='__main__':unittest.main()
