#!/bin/bash
# Live training monitor for V15 (grasp-center offset fix)
# Usage: ./monitor.sh [interval_seconds]
#
# This script periodically reads TensorBoard events and prints a
# compact summary of the key metrics.  Run it in a separate terminal.

INTERVAL="${1:-60}"
RUN_DIR="$(ls -td logs/skrl/cube_place/20* 2>/dev/null | head -1)"

if [[ -z "$RUN_DIR" ]]; then
    echo "No training run found."
    exit 1
fi

echo "Monitoring: $RUN_DIR"
echo "Refresh every ${INTERVAL}s  (Ctrl-C to stop)"
echo ""

while true; do
    conda run -n env_isaaclab python3 -u -c "
import os, sys
from tbparse import SummaryReader

run_dir = '$RUN_DIR'
reader = SummaryReader(run_dir)
df = reader.scalars
if len(df) == 0:
    print('  (no events yet)')
    sys.exit(0)

tags = [
    ('reach_fine',   'Info / Episode_Reward/reach_green_fine'),
    ('reach_coarse', 'Info / Episode_Reward/reach_green_coarse'),
    ('gripper',      'Info / Episode_Reward/gripper_phase'),
    ('cube_lift',    'Info / Episode_Reward/cube_lifted'),
    ('belt_hit',     'Info / Episode_Reward/belt_contact'),
    ('placed',       'Info / Episode_Reward/green_in_target'),
    ('grasp_rate',   'Info / Episode_Reward/metric_grasp_rate'),
    ('success',      'Info / Episode_Reward/metric_place_success'),
    ('reward',       'Reward / Total reward (mean)'),
    ('std',          'Policy / Standard deviation'),
    ('ep_len',       'Episode / Total timesteps (mean)'),
]

total = df[df['tag']=='Reward / Total reward (mean)'].sort_values('step')
steps = int(total.iloc[-1]['step'])
pct = steps / 5_000_000 * 100
ckpts = len([f for f in os.listdir(os.path.join(run_dir, 'checkpoints'))
             if f.startswith('agent_')])

import datetime
now = datetime.datetime.now().strftime('%H:%M:%S')
print(f'[{now}]  step={steps:>8,} ({pct:>5.1f}%)  ckpts={ckpts}')
for short, tag in tags:
    sub = df[df['tag']==tag].sort_values('step')
    if len(sub) == 0:
        continue
    v = sub.iloc[-1]['value']
    print(f'  {short:14s} {v:>10.4f}')
print()
" 2>/dev/null
    sleep "$INTERVAL"
done
