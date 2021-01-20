import autograd
import datetime
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
    trade_cutoff = datetime.date.today() - datetime.timedelta(14)
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


def get_payoffs(prices, option_price_indices, option_signs, option_strikes):
    # Returns a matrix of shape (n_options, n_paths)
    deltas = prices[:,option_price_indices] - option_strikes
    payoffs = numpy.maximum(deltas*option_signs, 0).T
    return payoffs


def get_loss(payoffs, options):
    loss = 0
    values = payoffs.mean(axis=1)
    for value, (t, date, bid, ask, strike, option_symbol) in zip(values, options):
        loss += numpy.maximum((bid - value)/bid, (value - ask)/ask)**2
    return loss / len(options)


def get_best_options(payoffs, options):
    model_values = {}
    real_values = {}
    best_options = []
    cutoff = datetime.date.today() + datetime.timedelta(365)  # keep 1y for long-term capital gain
    values = payoffs.mean(axis=1)
    for value, (t, date, bid, ask, strike, option_symbol) in zip(values, options):
        sign = {'call': 1, 'put': -1}[t]
        if value > 0 and date > cutoff:
            profit = value / ask - 1
            breakeven = strike + sign*ask  # what prices would have to move to for this option to pay for itself
            best_options.append((profit, option_symbol, ask, value, breakeven/current_price-1))
        model_values.setdefault(date, []).append(value)
        real_values.setdefault(date, []).append(ask)

    return model_values, real_values, best_options


def print_best_options(best_options, n=100):
    for profit, option_symbol, ask, value, breakeven_change in sorted(best_options, reverse=True)[:n]:
        print('%+24.2f%%: %s: ask %9.2f value %9.2f breakeven change %+9.2f%%' % (100. * profit, option_symbol, ask, value, 100.*breakeven_change))


def optimize_portfolio(payoffs, options, cash=50e3):
    option_asks = numpy.array([ask for t, date, bid, ask, strike, option_symbol in options])

    def f(portfolio):
        returns = cash + numpy.dot(portfolio, payoffs) - numpy.dot(portfolio, option_asks)
        return numpy.log(returns).mean()

    jac = autograd.grad(f)
    portfolio = numpy.zeros(len(options))

    while True:
        loss = f(portfolio)
        grad = jac(portfolio)
        grad[(grad < 0) & (portfolio == 0)] = 0  # Can't short options
        best_loss, best_portfolio = loss, portfolio
        for j in numpy.argsort(numpy.abs(grad))[-10:]:  # consider the 10 entries with largest absolute gradient
            new_portfolio = portfolio.copy()
            new_portfolio[j] += 100 * numpy.sign(grad[j])
            new_loss = f(new_portfolio)
            if new_loss > best_loss:
                best_loss, best_portfolio = new_loss, new_portfolio
        if best_loss <= loss:
            break
        loss, portfolio = best_loss, best_portfolio

    print('risk-adjusted expected outcome:', numpy.exp(loss))
    for p, (t, date, bid, ask, strike, option_symbol) in zip(portfolio, options):
        if p > 0:
            print(p, option_symbol, 'ask', ask)


def fit(symbol, current_price, options, alpha, beta, batch_n_paths=1000):
    print('fit(alpha=%f, beta=%f)' % (alpha, beta))
    date2i = get_dates(options)

    # Prepare some arrays so we can vectorize
    option_price_indices = []
    option_signs = []
    option_strikes = []
    for j, (t, date, bid, ask, strike, option_symbol) in enumerate(options):
        option_price_indices.append(date2i[date])
        option_strikes.append(strike)
        option_signs.append({'call': 1, 'put': -1}[t])

    def get_rvs(n_paths):
        return scipy.stats.levy_stable.rvs(alpha, beta, size=(n_paths, len(date2i)))

    def get_prices(params, rvs):
        loc, scale = params
        log_paths = numpy.cumsum(rvs*scale + loc, axis=1)
        log_paths = numpy.minimum(log_paths, 5)  # Just to avoid overflow
        prices = current_price * numpy.exp(log_paths)
        return prices

    def f(params, rvs):
        prices = get_prices(params, rvs)
        payoffs = get_payoffs(prices, option_price_indices, option_signs, option_strikes)
        loss = get_loss(payoffs, options)
        return loss

    jac = autograd.grad(f)

    # Fit the scale and location (volatility and drift)
    step_size = 1e-2
    params = numpy.array([-0.001, 0.033])
    adagrad_sum = 0.0
    for step in range(100):
        rvs = get_rvs(batch_n_paths)
        loc, scale = params
        print('%6d %12.6f %12.6f -> %24.6f' % (step, loc, scale, f(params, rvs)))
        grad = jac(params, rvs)
        adagrad_sum = adagrad_sum * 0.85 + numpy.dot(grad, grad)
        params -= step_size * grad / adagrad_sum**0.5
        step_size *= 0.95

    print('Generating lots of paths')
    rvs = get_rvs(40000)
    prices = get_prices(params, rvs)
    del rvs # free up memory
    payoffs = get_payoffs(prices, option_price_indices, option_signs, option_strikes)
    loss = get_loss(payoffs, options)
    model_values, real_values, best_options = get_best_options(payoffs, options)
    title = '%s: alpha = %f, beta = %f: loss = %f' % (symbol, alpha, beta, loss)
    print_best_options(best_options, 5)

    # Best options if prices start declining by x SDs annually
    number_sds = 1.0
    print('Annualized volatility:', params[1]*16)
    daily_extra_drift = -params[1]/16*number_sds
    print('Adding negative drift:', daily_extra_drift)
    prices_with_decline = prices * numpy.exp(daily_extra_drift * numpy.arange(prices.shape[1]))[None, :]
    payoffs_with_decline = get_payoffs(prices_with_decline, option_price_indices, option_signs, option_strikes)
    _, _, best_options_with_decline = get_best_options(payoffs_with_decline, options)
    print('If prices decline:')
    print_best_options(best_options_with_decline, 5)

    # Find best combination of options
    print('best portfolio:')
    optimize_portfolio(payoffs, options)

    print('best portfolio if prices decline:')
    optimize_portfolio(payoffs_with_decline, options)

    # return best_options, best_options_with_decline  # ignore plotting for now
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
    unique_strikes = sorted(set(date for _, date, _, _, _, _ in options))
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
        print('plotting', date)
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
    alpha, beta = 1.90, -0.95  # Looking at different stocks, this seems to fit the data reasonably well
    best_options, best_options_with_decline = fit(symbol, current_price, options, alpha, beta)
    all_best_options += best_options
    all_best_options_with_decline += best_options_with_decline

print('Best options:')
print_best_options(all_best_options)
print('Best options assuming some decline:')
print_best_options(all_best_options_with_decline)
