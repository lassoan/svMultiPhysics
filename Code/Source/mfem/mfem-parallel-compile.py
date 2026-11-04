"""Parallel compilation of the MFEM objects of the NMake build (NMake builds
serially; nvcc takes about 25 s per file). The compile commands are taken
from the generated makefiles; afterwards `cmake --build --target install`
links and installs with the up-to-date objects.

Run from the x64 Native Tools prompt with nvcc in the PATH (build-mfem.bat
sets this up), in the build directory."""
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

build = sys.argv[1]
jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
os.chdir(build)
flags = {}
for line in open('CMakeFiles/mfem.dir/flags.make'):
    m = re.match(r'^(\w+) = (.*)$', line.rstrip())
    if m:
        flags[m.group(1)] = m.group(2)
cmds = []
seen = set()
TAB = chr(9)
for line in open('CMakeFiles/mfem.dir/build.make'):
    line = line.rstrip()
    if not line.startswith(TAB) or ' -c ' not in line or '.obj' not in line:
        continue
    cmd = line.strip()
    mo = re.search(r' -o (\S+\.obj)', cmd)
    ms = re.search(r' -c (\S+) -o ', cmd)
    if not mo or not ms or mo.group(1) in seen:
        continue
    obj, src = mo.group(1), ms.group(1)
    seen.add(obj)
    if os.path.exists(obj) and os.path.getmtime(obj) > os.path.getmtime(src):
        continue
    for k, v in flags.items():
        cmd = cmd.replace('$(%s)' % k, v)
    cmds.append((obj, cmd))
print(len(cmds), 'objects to compile with', jobs, 'jobs', flush=True)
t0 = time.time()
failed = []


def run(item):
    obj, cmd = item
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    name = os.path.basename(obj)
    if r.returncode != 0:
        failed.append(obj)
        print('FAILED', obj, '\n', r.stdout[-3000:], r.stderr[-3000:], flush=True)
    else:
        print('%6.0f s  %s' % (time.time() - t0, name), flush=True)


with ThreadPoolExecutor(jobs) as ex:
    list(ex.map(run, cmds))
print('done in %.0f s, %d failed' % (time.time() - t0, len(failed)), flush=True)
sys.exit(1 if failed else 0)
