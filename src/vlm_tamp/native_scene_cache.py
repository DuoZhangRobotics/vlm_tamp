"""Bounded ownership of native collision indexes, separate from live queries."""
from collections import OrderedDict
import time

# A current observation and its immediate motion prediction are useful to reuse.
# Historical observations must not retain entire CAPT indexes until episode end.
MAX_CACHED_SCENES = 2


class NativeSceneCache(OrderedDict):
    def acquire(self,key,library,create):
        if key in self:
            self.move_to_end(key)
            return self[key],True
        # Release cache ownership before constructing another large index.
        # Native query clones hold shared_ptrs, so live queries remain valid.
        while len(self)>=MAX_CACHED_SCENES:
            _,(old_library,handle,_)=self.popitem(last=False)
            old_library.mcvamp_destroy(handle)
        started=time.perf_counter();handle=create()
        if not handle:raise RuntimeError('Native CAPT construction failed')
        entry=(library,handle,time.perf_counter()-started)
        self[key]=entry
        return entry,False

    def clear(self):
        while self:
            _,(library,handle,_)=self.popitem(last=False)
            library.mcvamp_destroy(handle)
