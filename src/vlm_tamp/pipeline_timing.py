"""Nested monotonic wall-clock spans; GPU stages explicitly synchronize."""
from contextlib import contextmanager
from datetime import datetime,timezone
import time
from .coupled_storage import write_json


class Timeline:
    def __init__(self):
        self.origin=time.perf_counter_ns();self.events=[];self.stack=[];self.marks=[];self.context={}
        self.started_utc=datetime.now(timezone.utc).isoformat()
    def now(self):return (time.perf_counter_ns()-self.origin)/1e9
    @staticmethod
    def sync_gpu():
        import sys
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():torch.cuda.synchronize()
    @contextmanager
    def span(self,name,*,gpu=False,**details):
        if gpu:self.sync_gpu()
        row={'id':len(self.events),'parent':self.stack[-1] if self.stack else None,'name':name,
             'start_s':self.now(),'gpu_synchronized':gpu,**self.context,**details}
        self.events.append(row);self.stack.append(row['id'])
        try:yield row
        except BaseException as error:
            row['error_type']=type(error).__name__;raise
        finally:
            if gpu:self.sync_gpu()
            row['end_s']=self.now();row['seconds']=row['end_s']-row['start_s'];self.stack.pop()
    def mark(self,name,**details):self.marks.append({'name':name,'at_s':self.now(),**self.context,**details})
    def call(self,name,fn,*,gpu=False):
        def wrapped(*args,**kwargs):
            with self.span(name,gpu=gpu):return fn(*args,**kwargs)
        return wrapped
    def save(self,path):
        write_json(path,{'schema':'pipeline-monotonic-timing-1','started_utc':self.started_utc,
            'events':self.events,'marks':self.marks,'elapsed_seconds':self.now(),
            'durations_are_inclusive':True,'nested_spans_must_not_be_added_together':True,
            'gpu_model_spans_synchronize_before_and_after':True})
