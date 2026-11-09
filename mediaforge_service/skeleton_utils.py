from __future__ import annotations
from fractions import Fraction
from pathlib import Path

def read_frames(path, *, output_fps=None, start=0.0, duration=None, max_frames=720):
    import av
    frames=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]; native=Fraction(stream.average_rate or stream.base_rate or 24)
        fps=Fraction(str(output_fps)).limit_denominator(100000) if output_fps else native
        interval=float(1/fps); end=start+duration if duration is not None else float('inf'); origin=None; previous=None; target=start
        for index, frame in enumerate(container.decode(stream)):
            timestamp=float(frame.time) if frame.time is not None else index/float(native)
            if origin is None: origin=timestamp
            timestamp-=origin
            if previous is not None:
                while target < min(timestamp,end)-1e-8:
                    if len(frames)>=max_frames: raise ValueError('Clip exceeds skeleton frame limit')
                    scale=min(1.0,768/max(frame.width,frame.height)); previous=previous.reformat(width=max(2,int(previous.width*scale)),height=max(2,int(previous.height*scale)),format='rgb24')
                    frames.append(previous.to_ndarray()); target+=interval
            if target>=end-1e-8: break
            previous=frame; previous_time=timestamp
        if previous is not None:
            target_shape = frames[0].shape[1::-1] if frames else (previous.width, previous.height)
            while target < min(previous_time+float(1/native),end)-1e-8:
                if len(frames)>=max_frames: raise ValueError('Clip exceeds skeleton frame limit')
                frames.append(previous.reformat(width=target_shape[0], height=target_shape[1], format='rgb24').to_ndarray()); target+=interval
    if len(frames)<2: raise ValueError('Selected video must contain at least two sampled frames')
    return frames,fps

def write_video(frames,fps,path):
    import av, numpy as np
    path=Path(path); temporary=path.with_suffix('.partial.mp4')
    with av.open(str(temporary),'w') as container:
        stream=container.add_stream('libx264',rate=fps); stream.width,stream.height=frames[0].shape[1],frames[0].shape[0]; stream.pix_fmt='yuv420p'; stream.options={'crf':'18','preset':'fast'}
        for i,pixels in enumerate(frames):
            frame=av.VideoFrame.from_ndarray(np.asarray(pixels,dtype=np.uint8),format='rgb24'); frame.pts=i; frame.time_base=1/fps
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    temporary.replace(path)
