.. desc:: Files with doctests to include in the pytest runner

.. doctest::

   >>> from importlib import util as importlib_util
   >>> def load_module(name, path):
   ...     module_spec = importlib_util.spec_from_file_location(name, path)
   ...     module = importlib_util.module_from_spec(module_spec)
   ...     module_spec.loader.exec_module(module)
   ...     return module
   >>> date = load_module('.', 'src/date/date.py')
   >>> from functools import partial
   >>> import doctest
   >>> testmod = partial(doctest.testmod, verbose=False, optionflags=4 | 8 | 32)

   date modules
   >>> _ = testmod(date); assert _.attempted != 0; assert _.failed == 0
