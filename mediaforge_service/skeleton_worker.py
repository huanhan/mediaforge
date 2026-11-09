from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from .skeleton_utils import read_frames, write_video
def run(c):
    import os
    try:
        import torch
        os.add_dll_directory(str(Path(torch.__file__).parent / 'lib'))
    except (ImportError, OSError):
        pass
    import sys as _sys; _sys.path.insert(0, c['runtime'])
    from mimicmotion.dwpose.dwpose_detector import DWposeDetector
    from mimicmotion.dwpose.util import draw_pose
    r=c['request']
    output_fps = r.get('output_fps')
    duration = r.get('duration_seconds')
    # Full-video requests with no explicit FPS are adapted to the bounded
    # in-memory frame window instead of failing halfway through decoding.
    if not output_fps and not duration:
        import av
        with av.open(c['source']) as probe:
            stream = probe.streams.video[0]
            native = float(stream.average_rate or stream.base_rate or 24)
            total = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else 0
        if total > 0 and total * native > 700:
            output_fps = max(1, min(native, 700 / total))
    frames,fps=read_frames(c['source'], output_fps=output_fps, start=r.get('start_seconds',0), duration=duration, max_frames=720)
    d=DWposeDetector(str(Path(c['models'])/'DWPose/yolox_l.onnx'),str(Path(c['models'])/'DWPose/dw-ll_ucoco_384.onnx'),c.get('device','cuda'))
    size=int(r.get('resolution',384)); scale=size/min(frames[0].shape[:2]); h=max(64,round(frames[0].shape[0]*scale/64)*64); w=max(64,round(frames[0].shape[1]*scale/64)*64)
    rendered=[]
    for i,frame in enumerate(frames):
        pose=d(frame); rendered.append(draw_pose(pose,h,w)); Path(c['progress']).write_text(str(.05+.9*(i+1)/len(frames)))
    d.release_memory(); write_video(np.stack(rendered).transpose(0,2,3,1),fps,c['output'])
if __name__=='__main__': run(json.loads(Path(sys.argv[1]).read_text()))
