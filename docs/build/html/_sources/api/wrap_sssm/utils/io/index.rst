wrap_sssm.utils.io
==================

.. py:module:: wrap_sssm.utils.io

.. autoapi-nested-parse::

   Helper functions for sssm (e.g. logger)

   ..
       !! processed by numpydoc !!


Attributes
----------

.. autoapisummary::

   wrap_sssm.utils.io.LOGGING_TYPES


Functions
---------

.. autoapisummary::

   wrap_sssm.utils.io.set_log_level


Module Contents
---------------

.. py:function:: set_log_level(verbose=None)

   
   Convenience function for setting the logging level.

   This function comes from the PySurfer package. See :
   https://github.com/nipy/PySurfer/blob/master/surfer/utils.py

   :param verbose: The verbosity of messages to print. If a str, it can be either
                   PROFILER, DEBUG, INFO, WARNING, ERROR, or CRITICAL.
   :type verbose: bool, str, int, or None















   ..
       !! processed by numpydoc !!

.. py:data:: LOGGING_TYPES

