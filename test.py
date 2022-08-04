import ray
import time

ray.init()

@ray.remote
def slow_function():
    time.sleep(5)
    return 1

import time
start = time.time()

refs = []
for _ in range(4):
    ref = slow_function.remote()
    refs.append(ref)

for r in refs:
    ray.get(r)

print(time.time() - start)