import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from vlm_tamp.region_multiview import SeparateRegionTools,request_arguments,separate_prompt
from vlm_tamp.rekep_libero.api import reservation_usd

def test_camera_inputs_are_exact_independent_panels(tmp_path,monkeypatch):
    from vlm_tamp.region_episode import RegionTools
    sheet=np.zeros((1120,1024,3),np.uint8)
    sheet[24:536,:512]=[20,40,60];sheet[24:536,512:]=[70,90,110]
    monkeypatch.setattr(RegionTools,'render_feedback',lambda self,path:Image.fromarray(sheet).save(path))
    tools=object.__new__(SeparateRegionTools);tools.last_collision=None
    tools.render_feedback(tmp_path/'feedback-sheet.png')
    args,files=request_arguments(tmp_path,'Use the current images.')
    pictures=[x for x in args['input'][0]['content'] if x['type']=='input_image']
    assert files==['camera-agentview.png','camera-frontview.png'] and len(pictures)==2
    for item,color in zip(pictures,([20,40,60],[70,90,110])):
        image=np.asarray(Image.open(BytesIO(base64.b64decode(item['image_url'].split(',',1)[1]))))
        assert image.shape==(512,512,3) and np.all(image==color)
    assert 'no shared camera-sheet' in separate_prompt()

def test_multi_image_reservation_counts_each_image_and_all_text(tmp_path):
    for name in ('agentview','frontview'):Image.new('RGB',(512,512)).save(tmp_path/f'camera-{name}.png')
    args,_=request_arguments(tmp_path,'x')
    content=args['input'][0]['content']
    text_bytes=sum(len(x['text'].encode()) for x in content if x['type']=='input_text')
    expected=((text_bytes+512+2*8192)*.125+32000*.5)/1e6
    assert reservation_usd(args)==pytest.approx(expected)
