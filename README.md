Put {stack, ROI} files in data/ with the following names

```
data/AMG#_exp#.{tif,zip}
```

Score on validation data:

```python
import numpy as np
from test import *

def classify(data):
    # in: image stack (time,height,width)
    # out: ROI masks (nroi,height,width)
    R = np.zeros((5, data.shape[0], data.shape[1]))
    # ...?
    return R

validation_data = ...
validation_labels = ...

print Score(classify, validation_data, validation_labels)
```

Score on test set (will not be possible on secret test set):

```python
import numpy as np
from test import *

def classify(data):
    # in: image stack (time,height,width)
    # out: ROI masks (nroi,height,width)
    R = np.zeros((5, data.shape[0], data.shape[1]))
    # ...?
    return R

T = TestSet()

print Score(classify, T.data, T.labels)
```

Score directly with predicted labels (will not be possible on secret test set):

```python
import numpy as np
from test import *

def classify(data, alpha, ...):
    # in: image stack (time,height,width)
    # out: ROI masks (nroi,height,width)
    R = np.zeros((5, data.shape[0], data.shape[1]))
    # ...?
    return R

validation_data = ...
vailidation_labels = ...
predicted_labels = classify(validation_data, 1, ...)

print Score(None, None, validation_labels, predicted_labels)
```

Score on function with multiple parameters:

```python
import numpy as np
from test import *

def classify(data, alpha, ...):
    # in: image stack (time,height,width)
    # out: ROI masks (nroi,height,width)
    R = np.zeros((alpha, data.shape[0], data.shape[1]))
    # ...?
    return R

validation_data = ...
validation_labels = ...

for alpha in range(1,5):
    print Score(lambda data: classify(data, alpha, ...), validation_data, validation_labels)
```
