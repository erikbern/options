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

symbol = sys.argv[1] if len(sys.argv) > 1 else 'TSLA'
ticker = yfinance.Ticker(symbol)
if ticker.info.get('bid') and ticker.info.get('ask'):
    current_price = (ticker.info['bid'] * ticker.info['ask'])**0.5
else:
    current_price = ticker.info['previousClose']
print('Current price: %f' % current_price)

options = []
trade_cutoff = datetime.date.today() - datetime.timedelta(7)
for date in ticker.options:
    print(date)
    data = ticker.option_chain(date)
    for t, df in [('call', data.calls), ('put', data.puts)]:
        for _, row in df.iterrows():
            if row['bid'] > 0 and row['ask'] > 0 and row['ask'] > row['bid'] and row['bid'] < 1.3 * row['ask'] and row['lastTradeDate'] > trade_cutoff:
                options.append((t, date, row['bid'], row['ask'], row['strike'], row['contractSymbol']))
unique_strikes = sorted(ticker.options)
print('Got %d options' % len(options))

# Generate all dates from now until last options
dates = []
dates_dates = []
day = datetime.date.today()
max_date = max(ticker.options)
while True:
    if day.weekday() < 5:
        dates.append(day.strftime('%Y-%m-%d'))
        dates_dates.append(day)
    day += datetime.timedelta(1)
    if day.strftime('%Y-%m-%d') > max_date:
        break
date2i = {date: i for i, date in enumerate(dates)}

def evaluate(prices, options):
    loss = 0
    model_values = {}
    real_values = {}
    largest_profit = 0
    best_options = []
    for t, date, bid, ask, strike, symbol in options:
        i = date2i[date]
        deltas = prices[:,i] - strike
        sign = {'call': 1, 'put': -1}[t]
        value = numpy.maximum(sign*deltas, 0).mean()
        mid = (bid * ask) ** 0.5
        loss_contribution = numpy.maximum((bid - value)/bid, (value - ask)/ask)**2
        loss += loss_contribution
        if value > 0:
            profit = numpy.log(value) - numpy.log(ask)
            if profit > largest_profit:
                largest_profit = profit
                best_options.append((profit, symbol, ask, value))
        model_values.setdefault(date, []).append(value)
        real_values.setdefault(date, []).append(mid)

    loss /= len(options)

    return loss, model_values, real_values, sorted(best_options, reverse=True)[:10]


def fit(alpha, beta, batch_n_paths=400, batch_n_options=400):
    print('fit(alpha=%f, beta=%f)' % (alpha, beta))

    def get_rvs(n_paths):
        return scipy.stats.levy_stable.rvs(alpha, beta, size=(n_paths, len(dates)))

    def get_prices(params, rvs):
        loc, scale = params
        log_paths = numpy.cumsum(rvs*scale + loc, axis=1)
        log_paths = numpy.minimum(log_paths, 5)  # Just to avoid overflow
        prices = current_price * numpy.exp(log_paths)
        return prices

    def f(params, rvs, options):
        return evaluate(get_prices(params, rvs), options)[0]

    jac = autograd.grad(f)

    # Fit the scale and location (volatility and drift)
    step_size = 1e-6
    params = numpy.array([-0.001, 0.033])
    for step in range(500):
        rvs = get_rvs(batch_n_paths)
        options_sample = random.sample(options, min(len(options), batch_n_options))

        loc, scale = params
        print('%6d %12.6f %12.6f -> %24.6f' % (step, loc, scale, f(params, rvs, options_sample)))
        params -= step_size * jac(params, rvs, options_sample)
        step_size *= 0.99

    rvs = get_rvs(40000)
    prices = get_prices(params, rvs)
    loss, model_values, real_values, best_options = evaluate(prices, options)
    for _, symbol, ask, value in best_options:
        print('%30s: ask %9.2f value %9.2f' % (symbol, ask, value))

    # Best options if prices start declining by 0.1% a day
    prices_with_decline = prices * numpy.exp(-0.001 * numpy.arange(prices.shape[1]))[None, :]
    _, _, _, best_options = evaluate(prices_with_decline, options)
    print('If prices decline:')
    for _, symbol, ask, value in best_options:
        print('%30s: ask %9.2f value %9.2f' % (symbol, ask, value))    

    # Plot average price over time
    pyplot.clf()
    pyplot.plot(dates_dates, numpy.mean(prices, axis=0), color=(1, 0, 0, 1), label='mean')
    pyplot.plot(dates_dates, numpy.median(prices, axis=0), color=(1, 0, 1, 1), label='median')
    pyplot.fill_between(dates_dates, numpy.percentile(prices, 2.5, axis=0), numpy.percentile(prices, 97.5, axis=0), color=(0, 0, 1, 0.2), label='95%')
    pyplot.fill_between(dates_dates, numpy.percentile(prices, 10, axis=0), numpy.percentile(prices, 90, axis=0), color=(0.5, 0, 1, 0.2), label='80%')
    pyplot.fill_between(dates_dates, numpy.percentile(prices, 25, axis=0), numpy.percentile(prices, 75, axis=0), color=(1, 0, 1, 0.2), label='50%')
    pyplot.legend()
    pyplot.savefig('avg_%.2f_%.2f.png' % (alpha, beta))

    # Plot distribution
    colors = pyplot.cm.viridis(numpy.linspace(0, 1, len(unique_strikes)))
    pyplot.clf()
    for date, color in zip(unique_strikes, colors):
        i = date2i[date]
        pyplot.plot(numpy.sort(prices[:,i]), numpy.linspace(0, 1, prices.shape[0], endpoint=False), color=color, alpha=0.8, label=date)
    pyplot.xlim([0.3 * current_price, 3.0 * current_price])
    pyplot.xscale('log')
    pyplot.legend()
    pyplot.savefig('distributions_%.2f_%.2f.png' % (alpha, beta))

    # Compute option prices under these paths
    pyplot.clf()
    for date, color in zip(unique_strikes, colors):
        pyplot.scatter(real_values[date], model_values[date], label=date, color=color, alpha=0.2)
    # pyplot.xlim([0, 1.5 * current_price])
    # pyplot.ylim([0, 1.5 * current_price])
    pyplot.legend()
    pyplot.xlabel('Real price')
    pyplot.ylabel('Model price')
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.title('alpha = %f, beta = %f' % (alpha, beta))
    pyplot.savefig('real_vs_model_%.2f_%.2f.png' % (alpha, beta))

    return loss

for alpha_beta in [(2, 0),
                   (1.75, 0), (1.75, -0.25),
                   (1.5, 0), (1.5, -0.25), (1.5, -0.5),
                   (1.0, 0), (1.0, -0.25), (1.0, -0.5), (1.0, -0.75)]:
    # 0 < alpha <= 2 is the stability. normal distribution = 2, cauchy = 1
    # -1 <= beta <= 1 is skewness. we constrain it to non-positive values
    alpha, beta = alpha_beta
    loss = fit(alpha, beta)
    print('%9.3f %9.3f -> %24.6f' % (alpha, beta, loss))
