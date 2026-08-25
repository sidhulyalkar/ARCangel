from pathlib import Path
import sys,json,hashlib,base64,re
EXPECTED_BUILD='S170-FINAL-20260824-C'
EXPECTED_NOTEBOOK_SHA='a2f64f68f092e7636b26d66fc482cb7c7de50b0530a695f6a049258a87ad72c1'
EXPECTED_SOURCE_SHA='60a425a86ea33239f41d0156ba17d55879cf1d814ff6822ec893cd71ab41cf19'

def main(path):
    p=Path(path); raw=p.read_bytes(); assert hashlib.sha256(raw).hexdigest()==EXPECTED_NOTEBOOK_SHA
    nb=json.loads(raw); text=json.dumps(nb)
    for m in [EXPECTED_BUILD,'PerceptualDecisionPolicy','ExplorationFrontier','V009 SYSTEMATIC FRONTIER PREFLIGHT PASS','V009 PERCEPTUAL SCIENTIST PREFLIGHT PASS','FRONTIER VALIDATION TELEMETRY','submission.parquet','VLLM_DISABLED_KERNELS','--limit-mm-per-prompt']:
        assert m in text,m
    assert 'MODEL_SCORE < 45' not in text
    source=''.join(nb['cells'][3]['source']); match=re.search(r'BUNDLE = "([^"]+)"',source); assert match
    bundle=base64.b64decode(match.group(1)); assert hashlib.sha256(bundle).hexdigest()==EXPECTED_SOURCE_SHA
    for i,c in enumerate(nb['cells']):
        if c.get('cell_type')=='code': compile(''.join(c.get('source',[])),f'cell-{i}','exec')
    print('S170 FINAL C NOTEBOOK VERIFICATION PASS')
if __name__=='__main__': main(sys.argv[1])
