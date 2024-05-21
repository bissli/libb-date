.. desc:: Files with doctests to include in the pytest runner

.. doctest::

   >>> from importlib import util as importlib_util
   >>> def load_module(name, path):
   ...     module_spec = importlib_util.spec_from_file_location(name, path)
   ...     module = importlib_util.module_from_spec(module_spec)
   ...     module_spec.loader.exec_module(module)
   ...     return module
   >>> date = load_module('.', 'src/date/date.py')
   >>> business = load_module('.', 'src/date/business.py')
   >>> from functools import partial
   >>> import doctest
   >>> from asserts import assert_equal as eq, assert_not_equal as ne
   >>> testmod = partial(doctest.testmod, verbose=False, optionflags=4 | 8 | 32)

   date modules
   >>> _ = testmod(date); ne(_.attempted, 0); eq(_.failed, 0)
   >>> _ = testmod(business); ne(_.attempted, 0); eq(_.failed, 0)
