import dask

import sys

package_name = 'dask'
if package_name in sys.modules:
    print(f"{package_name} is imported")
    print('version:',dask.__version__)
else:
    print(f"{package_name} is not imported")
