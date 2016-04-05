Put {stack, ROI} files in data/ with the following names

```
data/AMG#_exp#.{tif,zip}
```

Test in place:

```python
import numpy as np
from test import *

def classify(data):
    # in: image stack (time,height,width)
    # out: ROI masks (nroi,height,width)
    R = np.zeros((5, data.shape[0], data.shape[1]))
    # ...?
    return R

print Score(classify)
```

Test without reloading data:

```python
import numpy as np
from test import *

def classify(data, alpha):
    # in: image stack (time,height,width)
    # out: ROI masks (nroi,height,width)
    R = np.zeros((alpha, data.shape[0], data.shape[1]))
    # ...?
    return R

T = TestSet()

for alpha in range(5):
    print Score(lambda data: classify(data, alpha), test_set=T)
```
