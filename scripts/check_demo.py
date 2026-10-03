"""Exercise the live Gradio route, both examples, and validation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gradio_client import Client
from src.demo import resources
from src.config import RESULTS


def main():
    client=Client('http://127.0.0.1:7860',verbose=False)
    info=client.view_api(return_format='dict',print_info=False)
    named=info['named_endpoints']
    endpoint='/predict' if '/predict' in named else next(iter(named))
    df,_=resources()
    counts=df.groupby('allele').size().sort_values()
    checks=[]
    for allele in [counts.index[-1],counts.index[0]]:
        row=df[df.allele==allele].iloc[0]
        result=client.predict(row.peptide,allele,api_name=endpoint)
        assert allele in result[0] and 'not a calibrated' in result[0]
        table=result[1]
        assert len(table['data'])>=3
        checks.append(dict(allele=allele,peptide=row.peptide,arms=len(table['data']),passed=True))
    (RESULTS/'benchmark/demo_checks.json').write_text(json.dumps(dict(endpoint=endpoint,checks=checks,visual_check='Unavailable: no browser connection'),indent=2))
    print(checks)


if __name__=='__main__': main()
