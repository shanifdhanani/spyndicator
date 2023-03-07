from numpy import random
from sklearn import svm

from machine_learning.src.main.code.models.base_model import BaseModel
from machine_learning.src.main.code.models.model_types import ModelTypes


class SVMModel(BaseModel):
    """Epsilon-Support Vector Regression.

        The free parameters in the model are C and epsilon.

        The implementation is based on libsvm.

        Read more in the :ref:`User Guide <svm_regression>`.

        Parameters
        ----------
        C : float, optional (default=1.0)
            Penalty parameter C of the error term.

        epsilon : float, optional (default=0.1)
             Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
             within which no penalty is associated in the training loss function
             with points predicted within a distance epsilon from the actual
             value.

        kernel : string, optional (default='rbf')
             Specifies the kernel type to be used in the algorithm.
             It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
             a callable.
             If none is given, 'rbf' will be used. If a callable is given it is
             used to precompute the kernel matrix.

        degree : int, optional (default=3)
            Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels.

        gamma : float, optional (default='auto')
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            If gamma is 'auto' then 1/n_features will be used instead.

        coef0 : float, optional (default=0.0)
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.

        shrinking : boolean, optional (default=True)
            Whether to use the shrinking heuristic.

        tol : float, optional (default=1e-3)
            Tolerance for stopping criterion.

        cache_size : float, optional
            Specify the size of the kernel cache (in MB).

        verbose : bool, default: False
            Enable verbose output. Note that this setting takes advantage of a
            per-process runtime setting in libsvm that, if enabled, may not work
            properly in a multithreaded context.

        max_iter : int, optional (default=-1)
            Hard limit on iterations within solver, or -1 for no limit.

        pipeline (obj:`Pipeline`):
            If provided - the transformation pipeline that the model should use
    """

    ModelType = ModelTypes.Svm

    __slots__ = ()

    def __init__(self, kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, tol = 1e-3, C = 1.0,
                 epsilon = 0.1, shrinking = True, cache_size = 200, verbose = False, max_iter = -1,
                 pipeline = None):
        """
        Initialize the SVM model
        """

        super().__init__(pipeline = pipeline)

        self.model = svm.SVR(kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, tol = tol,
                             C = C, epsilon = epsilon, shrinking = shrinking, cache_size = cache_size,
                             verbose = verbose, max_iter = max_iter)
