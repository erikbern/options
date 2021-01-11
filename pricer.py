import autograd
import datetime
# import numpy
from autograd import numpy
from autograd.scipy.special import expit
import random
import scipy.optimize
import scipy.stats
import sys
import yfinance
from matplotlib import pyplot

def get_data(symbol):
    ticker = yfinance.Ticker(symbol)
    if ticker.info.get('bid') and ticker.info.get('ask'):
        current_price = (ticker.info['bid'] * ticker.info['ask'])**0.5
    else:
        current_price = ticker.info['previousClose']
    print('%s current price: %f' % (symbol, current_price))

    options = []
    trade_cutoff = datetime.date.today() - datetime.timedelta(7)
    for date in ticker.options:
        data = ticker.option_chain(date)
        date = datetime.date(*map(int, date.split('-')))
        for t, df in [('call', data.calls), ('put', data.puts)]:
            for _, row in df.iterrows():
                if row['contractSymbol'].startswith(symbol) and row['bid'] > 0 and row['ask'] > 0 and row['ask'] > row['bid'] and row['bid'] < 1.3 * row['ask'] and row['lastTradeDate'] > trade_cutoff:
                    options.append((t, date, row['bid'], row['ask'], row['strike'], row['contractSymbol']))

    print('%s number of options: %d' % (symbol, len(options)))
    return current_price, options


def get_dates(options):
    # Generate all dates from now until the last expiration
    dates = []
    day = datetime.date.today()
    max_date = max(date for _, date, _, _, _, _ in options)
    while True: 
        if day.weekday() < 5:
            dates.append(day)
        day += datetime.timedelta(1)
        if day > max_date:
            break
    date2i = {date: i for i, date in enumerate(dates)}
    return date2i


def evaluate(prices, options, date2i):
    loss = 0
    model_values = {}
    real_values = {}
    largest_profit = 0
    best_options = []
    for t, date, bid, ask, strike, option_symbol in options:
        i = date2i[date]
        deltas = prices[:,i] - strike
        sign = {'call': 1, 'put': -1}[t]
        value = numpy.maximum(sign*deltas, 0).mean()
        loss_contribution = numpy.maximum((bid - value)/bid, (value - ask)/ask)**2
        loss += loss_contribution
        if value > 0:
            profit = value / ask - 1
            if profit > largest_profit:
                largest_profit = profit
                best_options.append((profit, option_symbol, ask, value))
        model_values.setdefault(date, []).append(value)
        real_values.setdefault(date, []).append(ask)

    loss /= len(options)

    return loss, model_values, real_values, best_options


def print_best_options(best_options, n=100):
    for profit, option_symbol, ask, value in sorted(best_options, reverse=True)[:n]:
        print('%+24.2f%%: %s: ask %9.2f value %9.2f' % (100. * profit, option_symbol, ask, value))


def fit(symbol, current_price, options, alpha, beta, batch_n_paths=400, batch_n_options=400):
    print('fit(alpha=%f, beta=%f)' % (alpha, beta))
    date2i = get_dates(options)

    def get_rvs(n_paths):
        return scipy.stats.levy_stable.rvs(alpha, beta, size=(n_paths, len(date2i)))

    def get_prices(params, rvs):
        loc, scale = params
        log_paths = numpy.cumsum(rvs*scale + loc, axis=1)
        log_paths = numpy.minimum(log_paths, 5)  # Just to avoid overflow
        prices = current_price * numpy.exp(log_paths)
        return prices

    def f(params, rvs, options):
        return evaluate(get_prices(params, rvs), options, date2i)[0]

    jac = autograd.grad(f)

    # Fit the scale and location (volatility and drift)
    step_size = 1e-2
    params = numpy.array([-0.001, 0.033])
    adagrad_sum = 0.0
    for step in range(500):
        rvs = get_rvs(batch_n_paths)
        options_sample = random.sample(options, min(len(options), batch_n_options))

        loc, scale = params
        # print('%6d %12.6f %12.6f -> %24.6f' % (step, loc, scale, f(params, rvs, options_sample)))
        grad = jac(params, rvs, options_sample)
        adagrad_sum = adagrad_sum * 0.98 + numpy.dot(grad, grad)
        params -= step_size * grad / adagrad_sum**0.5
        step_size *= 0.99

    print('Generating lots of paths')
    rvs = get_rvs(40000)
    prices = get_prices(params, rvs)
    loss, model_values, real_values, best_options = evaluate(prices, options, date2i)
    title = '%s: alpha = %f, beta = %f: loss = %f' % (symbol, alpha, beta, loss)
    print_best_options(best_options, 5)

    # Best options if prices start declining by 1 SD annually
    print('Annualized volatility:', params[1]*16)
    daily_extra_drift = -params[1]/16
    print('Adding negative drift:', daily_extra_drift)
    prices_with_decline = prices * numpy.exp(daily_extra_drift * numpy.arange(prices.shape[1]))[None, :]
    _, _, _, best_options_with_decline = evaluate(prices_with_decline, options, date2i)
    print('If prices decline:')
    print_best_options(best_options_with_decline, 5)

    print('Plotting')

    # Plot average price over time
    pyplot.clf()
    dates = sorted(date2i.keys())
    pyplot.plot(dates, numpy.mean(prices, axis=0), color=(1, 0, 0, 1), label='mean')
    pyplot.plot(dates, numpy.median(prices, axis=0), color=(1, 0, 1, 1), label='median')
    pyplot.fill_between(dates, numpy.percentile(prices, 2.5, axis=0), numpy.percentile(prices, 97.5, axis=0), color=(0, 0, 1, 0.2), label='95%')
    pyplot.fill_between(dates, numpy.percentile(prices, 10, axis=0), numpy.percentile(prices, 90, axis=0), color=(0.5, 0, 1, 0.2), label='80%')
    pyplot.fill_between(dates, numpy.percentile(prices, 25, axis=0), numpy.percentile(prices, 75, axis=0), color=(1, 0, 1, 0.2), label='50%')
    pyplot.legend()
    pyplot.title(title)
    pyplot.savefig('%s_avg_%.2f_%.2f.png' % (symbol, alpha, beta))

    # Plot distribution
    unique_strikes = sorted(date for _, date, _, _, _, _ in options)
    colors = pyplot.cm.viridis(numpy.linspace(0, 1, len(unique_strikes)))
    pyplot.clf()
    for date, color in zip(unique_strikes, colors):
        i = date2i[date]
        pyplot.plot(numpy.sort(prices[:,i]), numpy.linspace(0, 1, prices.shape[0], endpoint=False), color=color, alpha=0.8, label=date)
    pyplot.xlim([0.3 * current_price, 3.0 * current_price])
    pyplot.xscale('log')
    pyplot.legend()
    pyplot.title(title)
    pyplot.savefig('%s_distributions_%.2f_%.2f.png' % (symbol, alpha, beta))

    # Compute option prices under these paths
    pyplot.clf()
    pyplot.plot([1e-2, 1e3], [1e-2, 1e3])
    for date, color in zip(unique_strikes, colors):
        pyplot.scatter(real_values[date], model_values[date], label=date, color=color, alpha=0.2)
    pyplot.legend()
    pyplot.xlabel('Real ask price')
    pyplot.ylabel('Model value')
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.title(title)
    pyplot.savefig('%s_real_vs_model_%.2f_%.2f.png' % (symbol, alpha, beta))

    return best_options, best_options_with_decline

all_best_options = []
all_best_options_with_decline = []
symbols = sys.argv[1:]

for symbol in symbols:
    current_price, options = get_data(symbol)
    # 0 < alpha <= 2 is the stability. normal distribution = 2, cauchy = 1
    # -1 <= beta <= 1 is skewness. we constrain it to non-positive values
    alpha, beta = 1.9, -0.95  # Looking at different stocks, this seems to fit the data reasonably well
    best_options, best_options_with_decline = fit(symbol, current_price, options, alpha, beta)
    all_best_options += best_options
    all_best_options_with_decline += best_options_with_decline

print('Best options:')
print_best_options(all_best_options)
print('Best options assuming a 1SD decline:')
print_best_options(all_best_options_with_decline)
