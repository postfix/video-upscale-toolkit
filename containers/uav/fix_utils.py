"""Patch UAV's read_frame_from_videos to use cv2 instead of torchvision.io.

torchvision.io.read_video → PyAV → swscale breaks on real-world videos in
this container with "Resource temporarily unavailable" (EAGAIN from swscale
init), regardless of color tags. decord (the other obvious replacement) is
worse — segfaults on `get_batch()` in our environment.

cv2.VideoCapture is the stable path. UAV already uses it in the folder
branch of the same function, so semantics are identical to existing code.
"""
import re
import sys

PATH = "/opt/Upscale-A-Video/utils.py"

NEW_FUNC = '''def read_frame_from_videos(frame_root):
    # PATCH: cv2.VideoCapture instead of torchvision.io.read_video.
    # PyAV+swscale fails ("EAGAIN") on many real-world inputs in this image;
    # decord segfaults at get_batch. cv2 is what the folder branch already uses.
    import numpy as _np
    if frame_root.endswith(VIDEO_EXTENSIONS):
        video_name = os.path.basename(frame_root)[:-4]
        _cap = cv2.VideoCapture(frame_root)
        fps = float(_cap.get(cv2.CAP_PROP_FPS)) or None
        _frames = []
        while True:
            _ret, _f = _cap.read()
            if not _ret:
                break
            _frames.append(_f[..., [2, 1, 0]])  # BGR -> RGB
        _cap.release()
        if not _frames:
            raise RuntimeError(f"cv2 read 0 frames from {frame_root}")
        frames = torch.from_numpy(_np.ascontiguousarray(_np.array(_frames))).permute(0, 3, 1, 2).contiguous()
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))[..., [2, 1, 0]]
            frames.append(frame)
        fps = None
        frames = torch.Tensor(_np.array(frames)).permute(0, 3, 1, 2).contiguous()
    size = frames[0].size
    return frames, fps, size, video_name
'''

with open(PATH) as f:
    src = f.read()

new_src, n = re.subn(
    r'def read_frame_from_videos.*?return frames, fps, size, video_name\n',
    NEW_FUNC,
    src,
    count=1,
    flags=re.DOTALL,
)

if n != 1:
    print(f"fix_utils: expected to replace 1 function definition, replaced {n}", file=sys.stderr)
    sys.exit(1)

# Make sure numpy is imported at module level so the inner imports are cheap
if "\nimport numpy" not in new_src and "\nimport numpy as " not in new_src:
    new_src = new_src.replace("import torchvision\n", "import torchvision\nimport numpy as np\n", 1)

with open(PATH, "w") as f:
    f.write(new_src)

print("utils.py: read_frame_from_videos now uses cv2.VideoCapture")
