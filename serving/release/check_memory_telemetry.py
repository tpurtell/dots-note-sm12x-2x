#!/usr/bin/env python3
"""CPU parser compatibility for historical/current GPU telemetry."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from report import memory_summary
with TemporaryDirectory() as directory:
    root=Path(directory);attempt=root/'attempt-test';attempt.mkdir()
    samples=['0, GPU-a, 100, 200, 50, 90',
             '0, GPU-a, 120, 200, 60, 95, 79, 1800, 13000, Not Active, Active',
             '0, GPU-a, 110, 200, 55, 91, N/A, N/A, [N/A], N/A, Not Active',
             '0, GPU-a, 115, 200, 50, 93, 81, 1750, 13000']
    (attempt/'memory.jsonl').write_text('\n'.join(json.dumps({'time':i,'returncode':0,'gpu_csv':sample,'meminfo':'MemAvailable: 100 kB'}) for i,sample in enumerate(samples)))
    result=memory_summary(root,'rtx');assert not result['monitor_errors'];gpu=result['hosts']['rtx']['gpus']['GPU-a']
    assert gpu['memory_used_mib']['samples']==4 and gpu['peak_observed_memory_used_mib']==120
    assert gpu['optional_telemetry']['temperature_celsius']['samples']==2
    assert gpu['optional_telemetry']['temperature_celsius']['max']==81
    assert gpu['thermal_slowdown']['hw_thermal_slowdown']=={'observations':2,'active_samples':1}
    assert gpu['thermal_slowdown']['sw_thermal_slowdown']=={'observations':1,'active_samples':0}
print('PASS historical6/current9/11 columns, optional N/A and thermal flags')
