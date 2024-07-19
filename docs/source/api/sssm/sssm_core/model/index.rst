sssm.sssm_core.model
====================

.. py:module:: sssm.sssm_core.model


Classes
-------

.. autoapisummary::

   sssm.sssm_core.model.Config
   sssm.sssm_core.model.Context_Cont_configs
   sssm.sssm_core.model.Model
   sssm.sssm_core.model.TC
   sssm.sssm_core.model.augmentations
   sssm.sssm_core.model.base_Model


Module Contents
---------------

.. py:class:: Config

   Bases: :py:obj:`object`


.. py:class:: Context_Cont_configs

   Bases: :py:obj:`object`


.. py:class:: Model(device='cuda', model_name='model.pt', model_path=None)

   Bases: :py:obj:`sklearn.base.ClassifierMixin`, :py:obj:`sklearn.base.BaseEstimator`


   .. py:method:: __load_model(model_path=None, model_name='model.pt')


   .. py:method:: __predict(data)


   .. py:method:: __rearrange_output(proba, fea, n_epoch)


   .. py:method:: __sliding_window_sample(X=None, step=50)


   .. py:method:: plot_predictions(epoch_ind=0)


   .. py:method:: predict(X=None, step=50)

      
      input data, predict sleep event for each time step
      :param X: (n_epoch,  n_time)
      :param step: 300 >= step >= 1
      :return: predict_proba, pred, filtered pred, feature
















      ..
          !! processed by numpydoc !!


   .. py:method:: predict_proba(X=None, step=50)

      
      input data, predict sleep event for each time step
      :param X: (n_epoch,  n_time)
      :param step: 300 >= step >= 1
      :return: predict_proba, pred, filtered pred, feature
















      ..
          !! processed by numpydoc !!


   .. py:method:: to_json()


   .. py:method:: to_pandas(overall_threshold=0.5, describe=False, event_threshold=None)


.. py:class:: TC

   Bases: :py:obj:`object`


.. py:class:: augmentations

   Bases: :py:obj:`object`


.. py:class:: base_Model(configs)

   Bases: :py:obj:`torch.nn.Module`


   
   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool















   ..
       !! processed by numpydoc !!

   .. py:method:: forward(x_in)


