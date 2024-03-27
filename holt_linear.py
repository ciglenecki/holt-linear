class HoltLinear:
    """
    Second-order exponential smoothing (Holt linear) with the limitation of a starting trend of 0.
    https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing_(Holt_linear)

    Simple exponential smoothing does not do well when there is a trend in the data. In such situations, second-order exponential smoothing is applied, which is the recursive application of an exponential filter twice.

    The basic idea behind double exponential smoothing is to introduce a term to take into account the possibility of a series exhibiting some form of trend. This slope component (b) is itself updated via exponential smoothing.

    Initialization:
        s_0 = x_0
        b_0 = x_1 - x_0

    Updates for t > 0:
        s_t = alpha * x_t + (1 - alpha) * (s_{t-1} + b_{t-1})
        b_t = beta * (s_t - s_{t-1}) + (1 - beta) * b_{t-1}

    Forecast for m > 0 steps:
        f_{t+m} = s_t + b_t * m


    Equivalent using `statsmodels` package:

        initial_values = [5,2]

        # first iter:
        holt_1 = Holt(initial_values)
        fit1 = holt_1.fit(
            smoothing_level=alpha,
            smoothing_trend=beta
        )

        # second iter:
        # stupid statsmodels doesn't allow update with a single value
        # so we have to hack it by using the last two values

        new_value = 4
        holt2_values = np.array([holt_1.endog[-1], new_value])
        holt2 = Holt(
            holt2_values,
            initialization_method='known',
            initial_level=fit1.level[-2],
            initial_trend=fit1.trend[-2],
        )
        fit2 = holt2.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
        )
        fit2.fittedvalues
        steps = 1
        fit2.forecast(steps)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0):
        """
        Args:
            alpha: controls the smoothing of the level
            beta: controls the smoothing of the trend
        """

        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1

        self.alpha = alpha
        self.beta = beta
        self.s: float | None = None
        self.s_prev: float | None = None
        self.b: float = 0
        self.x_prev: float | None = None

    def get_smoothed_value(self) -> float:
        """s_t + b_t"""

        assert self.s is not None
        return self.s + self.b

    def forecast(self, steps: float = 1) -> float:
        """f_{t+m} = s_t + b_t * m"""

        assert self.s is not None
        return self.s + self.b * (steps)

    def update(self, x: float) -> float:

        if self.s is None:
            # Initialization of s (First x observation)
            self.s = x
            self.x_prev = x
            return x

        s_next = self._calc_next_s(x)
        self.s_prev = self.s
        self.s = s_next
        self.b = self._calc_next_b()
        self.x_prev = x
        return self.get_smoothed_value()

    def _calc_next_s(self, x: float) -> float:
        """s_t = alpha * x_t + (1 - alpha) * (s_{t-1} + b_{t-1})"""

        assert self.s is not None
        b = 0 if self.b is None else self.b
        first_term = self.alpha * x
        second_term = (1 - self.alpha) * (self.s + b)
        s_next = first_term + second_term
        return s_next

    def _calc_next_b(self) -> float:
        """b_t = beta * (s_t - s_{t-1}) + (1 - beta) * b_{t-1}"""

        assert self.s is not None
        assert self.s_prev is not None
        first_term = self.beta * (self.s - self.s_prev)
        second_term = (1 - self.beta) * self.b
        b_next = first_term + second_term
        return b_next

    def __getstate__(self) -> dict[str, float | None]:
        """Used for pickling"""

        return dict(
            alpha=self.alpha,
            beta=self.beta,
            s=self.s,
            s_prev=self.s_prev,
            b=self.b,
            x_prev=self.x_prev,
        )

    def __setstate__(self, new_state) -> None:
        """Used for pickling"""
        self.__dict__.update(new_state)

    def __repr__(self) -> str:
        """Get string representation of state but reduce decimal points for float."""

        str_dict = {
            k: "{:.3f}".format(v) if isinstance(v, float) else v
            for k, v in self.__getstate__().items()
        }
        return f"{self.__class__.__name__} {str_dict}"
