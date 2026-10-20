"""Replay saved sensor inputs through both public geometry entry points.

This is a read-only component check. No robot motion or model call is made,
and no saved regions or computed anchors are supplied to the acting agent.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
import reliable_microwave_final as scripted
import serve_subscription_robot as agent
from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.hinged_contact_tools import tool_configuration,tool_contract


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for state in [1,11,21,31]:
        episode=ROOT/f'results/frozen-prepared-method-transfer-20261020/state{state:03d}/attempt01'
        folder=sorted(episode.glob('*-grounding'))[0]
        choices=read_json(folder/'agent-regions.json');spec=read_json(folder/'model.json')
        metadata=read_json(episode/'opened/metadata.json')
        cameras,_=read_observation_v3(episode/'opened',metadata)
        left=scripted.ground_handle(cameras,choices,spec,metadata)
        right=agent.ground_handle(cameras,choices,spec,metadata)
        assert left[0].spec==right[0].spec
        for a,b in zip(left[1:4],right[1:4]):np.testing.assert_array_equal(a,b)
        assert left[4]==right[4]
        low,high=right[4]['pad_support']['observed_support_z_quantiles_m']
        span=right[4]['pad_support']['pad_vertical_span_bound_m']
        margin=right[4]['pad_support']['support_margin_m']
        assert all(low+span+margin<=point[2]<=high-span-margin for point in right[3])
        rows.append({'state':state,'sensor_source':str(episode/'opened'),
            'identical_cloud_mask_anchors_and_model':True,'anchor_count':len(right[3]),
            'pad_support':right[4]['pad_support']})
    names=['ground_handle','select_grasp','contact_plan','NativeScene','TaskIK']
    assert all(getattr(agent,n) is getattr(scripted,n) for n in names)
    write_json(out/'parity.json',{'cases':rows,'same_callable_objects':names,
        'tool_contract':tool_contract(tool_configuration()),
        'robot_motion':False,'model_calls':0,'historical_inputs_given_to_agent':False})
    print(json.dumps({'matched_geometry_cases':len(rows),'shared_functions':names}),flush=True)


if __name__=='__main__':
    main()
