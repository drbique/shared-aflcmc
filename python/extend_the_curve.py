'''
 Filename: extend_the_curve.py
 Purpose: Provide the functionality of extend_the_curve() as follows:

 Inputs:
    y                          data (dependent variable)
    desired_forecast_periods   desired number of intervals/periods in forecast
    interval                   constant width on x-axis between adjacent points, which should always be specified
                               whenever same units are used on both axes.
                               interval >= 0; interval == 0 (default) => use average change in y
                               whenever default yields unsatisfactory forecast, try non-zero interval (e.g. interval=1.0)
    origin                     x[0]
    returns                    one of the following
                                   "step" for an integer step-function result
                                   "line" for the line segment which is the extension of the curve (default)
                                   "slope" for slope and intercept only
                                   "length" for for slope and intercept with suggested length of forecast (periods)
                                   "flag" for slope and intercept with length and flag (See below)
    epsilon                    estimate of 'machine' or computational error
                               (normally omit this optional parameter)                            

 Require: 1 < desired_forecast_periods

 Returns - Depend on return argument, which defaults to "line" 
    "flag"
        Returns m,i,p,flag
            m and i denote the slope and intercept, respectively, of the desired line
            p denotes the number of valid periods for forecast
            flag  < 1 indicates forecast is proportionately less reliable for desired period, 
            flag >= 1 indicates forecast if proportionately more reliable for desired period

    "length"
        Returns m,i
            m and i denote the slope and intercept, respectively, of the desired line, and
            suggested number of periods in forecast, which is the suggested length of the predicted line

    "slope"
        Returns m,i
            m and i denote the slope and intercept, respectively, of the desired line

    "step"
        Returns z
            z is the extended line segment or step-function rounding results to integers

    "line"
        Returns z
            z is the extended line segment

(C) COPYRIGHT NOTICE

All or portions of the documentation and software included in this software
distribution from AFLCMC/HNII are copyrighted by Stephen Bique, who has assigned
All Rights for those portions to AFLCMC/HNII.  Outside the USA, AFLCMC/HNII has
copyright on some or all of the software developed for AFLCMC/HNII. Any files may
contain specific copyright notices and those notices must be retained in any derived
work.

AFLCMC/HNII LICENSE

AFLCMC/HNII may grant permission for redistribution and use in source and binary
forms, with or without modification, of this software and documentation
created for AFLCMC/HNII provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgements:

   This product includes software developed for AFLCMC/HNII.

4. Neither the name of AFLCMC/HNII nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THE SOFTWARE PROVIDED BY AFLCMC/HNII IS PROVIDED BY AFLCMC/HNII AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL AFLCMC/HNII OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation
are those of the authors and should not be interpreted as representing
official policies, either expressed or implied, of AFLCMC/HNII.
'''


def extend_the_curve(y, desired_forecast_periods=14, interval=0.0, origin=0.0, returns="line", epsilon = 0.0000005):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    import re

    y = np.asarray(y)

    '''
     Seek a method that in some sense is not too sensitive to slight changes in the data.
     So we seek to obtain mostly consistent answers with slightly perturbed data.

     As we look larger and larger samples, we expect the variance to become less,
     approaching the variance of the noise that is added. However, there is an inherent
     problem: for the smaller samples, we expect higher variance, which makes a fair
     comparison more difficult.
    '''
    # Get average change in y:
    scale = np.average([abs(v - y[j]) for j, v in enumerate(y[1:])])

    # Check interval width and set if invalid
    if interval <= epsilon:
        interval = scale # use average change in y
        if interval <= epsilon:
            interval = 1.0

    # Define x
    x = np.asarray([j * interval + origin for j in range(len(y))])

    # Set reasonable scale for noise to perturb values of y:
    scale = max(0.00015, scale * 0.25)

    '''
    Unless change in recent data indicates possibly zero slope, start with six points; otherwise, four points.
    Assuming errors in the data, the recent slopes may be invalid. Before we suspect zero slope, we expect
    either the slope is constant (no change in the data) or slopes are approaching zero.
    If the angles in degrees decrease approximately from 47 to 32 to 15, we conservatively guess slope might
    be tending to zero.
    '''
    slope0 = abs(y[-3] - y[-4]) / interval
    slope1 = abs(y[-2] - y[-3]) / interval
    slope2 = abs(y[-1] - y[-2]) / interval

    # print('Recent slopes:', slope0, ' ', slope1, ' ', slope2)

    if (slope0 < 1.07237) and (slope1 < 0.62487) and (slope2 < 0.26795):
        start = 4
    else:
        start = 6

    while True:
        hs = []
        for _ in range(373):
            # round

            # Fix perturbed y for this round
            y_noisy = y + np.random.normal(0.0, scale, size=len(y))

            # Initializations
            first_time = True
            rmse0 = 0.0
            h_best = max(6, desired_forecast_periods)
            size = h_best - start + 1
            h = start - 1
            for _ in range(size):
                h1 = h
                h = h1 + 1

                first = len(x) - h1  # use points first..last, inv. fixed on ea. loop execution

                X = x[first:]
                Y = y_noisy[first:]

                slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

                rmse = sum([abs(slope * X[j] + intercept - val) for j, val in enumerate(Y)]) / float(len(X))

                if first_time:
                    rmse0 = rmse
                    if rmse < epsilon:
                        f = 0.255
                    else:
                        f = rmse * 0.255
                    first_time = False
                elif rmse - rmse0 > f:

                    h_best = h - 1
                    break

            hs.append(h_best)

        # Count frequencies of mode and next most frequently occurring value, possibly the same frequency
        freq = {}  # Creating an empty dictionary
        for item in hs:
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1
        f1 = 0
        f2 = 0
        for key, value in freq.items():
            if value >= f1:
                f2 = f1
                f1 = value
                m = key
            elif value >= f2:
                f2 = value
        if (f1 - f2 > 22):

            # Determine weights for linear regression
            m1 = m + 1

            # Suppose there is a small absolute error in m.
            # If m is fairly large, then such small error is likely
            # not nearly as 'bad' as when m is small.
            # When m is small, there is a good chance that
            # the best m is actually smaller. Why? Because whenever
            # there is a maxima or minima, the best regression line
            # minimizes the computed error but deviates from the tangent.
            # In particular, if the curve is leveling off, the above algorithm
            # will likely find m=4 as that is the smallest possible.
            # When m is large, it is likely the error will be small
            # if a smaller value is used.
            # With the above justification, we normally give more
            # weight to points closer to the last observation.
            # However, if the function behaves like a step-function,
            # or if there are significant changes in the slope, then we
            # conservatively apply equal weights to all points.

            changes = 0
            last = y[-m1]
            last_change = y[-m] - last
            if abs(last_change) < epsilon:
                no_times_constant = 1
            else:
                no_times_constant = 0
            for value in y[-m + 1:]:
                change = value - last
                if ((last_change >= 0) != (change >= 0)):
                    changes += 1
                last_change = change
                if abs(change) < epsilon:
                    no_times_constant += 1
                last = value

            # Does the function behave partly as a step-function?
            if no_times_constant >= max(1, m * 0.2):
                # If the function is a step-function, our regression approach
                # may have underestimated m.
                m_ext = desired_forecast_periods - m
                if m < desired_forecast_periods:
                    last = y[-desired_forecast_periods]
                    no_times = 0
                    for value in y[-desired_forecast_periods+1:-m]:
                        if abs(value - last) < epsilon:
                            no_times += 1
                        last = value
                    if no_times >= max(1, m_ext * 0.2):
                        m = desired_forecast_periods # adjust m

                use_adj_weights = False

            elif (m <= 4):
                '''
                  Consider   'actual' m      predicted m   'effectve' m using assigned weights
                                 1               4                      2
                                 2               4                      2  
                                 3               4                      2
                                 4               4                      2
                  If we suppose that m's in the range 1-4 are more likely < 4, then by using the 'effective' m's,
                  we will be assigning better weights in view of the 'actual' m's. The supposition is reasonable 
                  since there is no reason to claim the m's are not approximately uniformly distributed.
                '''
                use_adj_weights = True  # linearly adjust keeping sum of weights equal to m
            elif changes >= max(1, m * 0.125) or m < max(14, desired_forecast_periods):
                # The function's slope changes sign significantly, or
                # m is small enough that the recent points are weighed sufficiently heavily.
                use_adj_weights = False
            else:
                use_adj_weights = True # linearly adjust keeping sum of weights equal to m

            w = [1.0] * m   # start with uniform weights
            first = len(x) - m

            X = x[first:]
            Y = y[first:]

            # sckit-learn implementation
            # Use only one feature
            u = X.reshape(-1, 1)
            v = Y.reshape(-1, 1)

            # Model fit
            regression_model = LinearRegression().fit(u, v, w)
            slope = regression_model.coef_[0][0]
            # print('slope = {0}'.format(slope))
            # Adjust line to go through last point
            intercept = Y[-1] - slope * X[-1]

            if use_adj_weights:
                R0 = [pow(x * slope + intercept - y, 2) for x, y in zip(X,Y)]
                a = 0.0  # left bound for weight factor
                b = 1.0  # right bound for weight factor

                # Assign weights with higher values for points closer to the rightmost observation
                wsum = 2.0 / float(m1)
                w1 = np.asarray([v * wsum for v in range(1, m1, 1)])
                w0 = np.asarray([1.0] * m)

                # Apply bisection algorithm to search for best weight factor to linearly combine w0 & w1
                while (b-a) >= 0.01:
                    fact = (a + b) / 2

                    wc = 1.0 - fact
                    w = w1 * fact + w0 * wc

                    # Model fit
                    regression_model = LinearRegression().fit(u, v, w)
                    slope_ = regression_model.coef_[0][0]

                    # Adjust line to go through last point
                    intercept_ = Y[-1] - slope * X[-1]

                    R = [pow(x * slope_ + intercept_ - y, 2) for x, y in zip(X, Y)]
                    if R < R0:
                        R0 = R
                        slope = slope_
                        intercept = intercept_
                        a = fact
                    else:
                        b = fact

            returns = re.sub(r'[^a-z\d ]','',returns.lower())
            if ("length" in returns) or ("flag" in returns):
                # Return additional values (m and flag), mainly for testing purposes.
                # To ignore extra values, use slope,intercept,_,_ = extend_the_curve(y)

                # Check m by recalculating using given data and computed line
                rsum = 0.0
                for j in [-4, -3, -2, -1]:
                    rsum += abs(slope * x[j] + intercept - y[j])
                m_best = 4
                eps = epsilon * 4.0
                rbest = rsum / 4.0
                m_max = min(desired_forecast_periods * 2 + 3, len(x))
                for j in range(-5, -m_max, -1):
                    rsum += abs(slope * x[j] + intercept - y[j])
                    r = rsum / float(-j)
                    eps += epsilon
                    if r <= rbest + epsilon:
                        m_best = -j

                if m < m_best:
                    m = m_best
                if "flag" in returns:
                    return (slope, intercept, m, m / desired_forecast_periods)
                else:
                    return (slope, intercept, m)

            elif "slope" in returns:
                # Alternatively, just return slope and intercept:
                return (slope, intercept)
            else:
                # Alternatively, return predicted line segment:
                val = x[-1]
                result = []
                if "step" in returns:
                    for index in range(desired_forecast_periods):
                        val += interval
                        result.append(round(slope * val + intercept))
                else:
                    for index in range(desired_forecast_periods):
                        val += interval
                        result.append(slope * val + intercept)
                return result