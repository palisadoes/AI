'''Module to analyse ETL data.'''

from statsmodels.tsa.stattools import adfuller


def stationary(values):
    """Evaluate wether the timeseries is stationary.

    non-stationary timeseries are probably random walks and not
    suitable for forecasting.

    Args:
        values: Pandas series of values on which to evaluate stationarity.

    Returns:
        state: True if stationary

    """
    # Initialize key variables
    state = False
    criteria = []

    # statistical test
    result = adfuller(values)
    adf = result[0]
    print('> Stationarity Test:')
    print('  ADF Statistic: {:.3f}'.format(adf))
    print('  p-value: {:.3f}'.format(result[1]))
    print('  Critical Values:')
    for key, criterion in result[4].items():
        print('\t{}: {:.3f}'.format(key, criterion))
        criteria.append(criterion)

    # Return
    if adf < min(criteria):
        state = True
    print('  Stationarity: {}'.format(state))
    return state
