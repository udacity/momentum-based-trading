import numpy as np
import csv
import sqlite3
from scipy.stats import norm
from contextlib import closing

conn = None
cs = None

class GBM:
    def __init__(self):
        self.mu = np.nan;
        self.sigma = np.nan;
        self.rng = np.random.default_rng()
        
    def simulate(self, N, K, Dt, S0):
        sqrt_Dt = np.sqrt(Dt)
        traj = np.full((N+1, K), np.nan)
        drift = (self.mu - self.sigma**2/2) * np.linspace(1, N, N) * Dt
        for i in range(K):
            W = sqrt_Dt * np.cumsum(norm.rvs(size=N))
            traj[1:, i] = S0 * np.exp(drift + self.sigma * W)
            traj[0, i] = S0
        return traj;

    def calibrate(self, trajectory, Dt):
        increments = np.diff(np.log(trajectory));
        moments = [0, 0];
        n_iter = 10;
        for iter in range(n_iter):
            X = self.rng.choice(increments, size=len(increments)//2)
            moments[0] += np.mean(X)/n_iter;
            moments[1] += np.mean(X**2)/n_iter
        std = np.sqrt(moments[1] - moments[0]**2);
        self.sigma = std/np.sqrt(Dt);
        self.mu = moments[0] / Dt + self.sigma**2/2;

    def forecast(self, latest, T, confidence):
        m = (self.mu - self.sigma**2/2)/2 * T;
        s = self.sigma * np.sqrt(T);
        Q = norm.ppf([(1-confidence)/2, (1+confidence)/2], loc=m, scale=s)
        return {
            'confidence': confidence,
            'expected': latest * np.exp(m),
            'interval': latest * np.exp(Q)
        };

    def expected_shortfall(self, T, confidence):
        m = (self.mu - self.sigma**2/2)/2 * T;
        s = self.sigma * np.sqrt(T);
        ES = -m + s * norm.pdf(norm.ppf(confidence))/(1 - confidence);
        return ES;


def prepare():
    cs.execute("""
    create table if not exists prices (
    theday text primary key,
    price real
    );
    """)
    with closing(open('../SP500.csv')) as datafile:
        reader = csv.DictReader(datafile, fieldnames=["date", "price"], delimiter='\t')
        for row in reader:
            cs.execute(F"""
            insert or ignore into prices values (\"{row['date']}\",
            {float(row['price'])});
            """)
    cs.execute(F"""
    create table if not exists positions (
    time_of_trade text,
    instrument text,
    quantity real,
    cash real,
    primary key (time_of_trade, instrument)
    );
    """);
    cs.execute(F"""
    insert or ignore into positions values
    ('1666-01-01', 'SP500', 0, 1000000);
    """)
    conn.commit()

def position_size(which_day, forecast, ES):
    cs.execute(F"""
    select quantity, cash from positions
    where instrument = 'SP500'
    and time_of_trade < '{which_day}'
    order by time_of_trade desc
    limit 1;
    """)
    qty, cash = cs.fetchall()[0]
    cs.execute(F"""
    select price from prices
    where theday <= '{which_day}'
    order by theday desc
    limit 1;
    """);
    price = cs.fetchall()[0][0]
    if price < forecast['interval'][0]:
        return qty + round(cash/price)
    elif price > forecast['interval'][1]:
        return -qty
    else:
        return qty


def analyse(which_day):
    cs.execute(F"""
    select price from prices where theday <= '{which_day}'
    order by theday desc limit 120;
    """)
    P = np.flipud(np.asarray(cs.fetchall())).flatten();
    model = GBM();
    Dt = 1.0/250;
    model.calibrate(P, Dt);
    confidence = 0.95
    T = 10 * Dt;
    forecast = model.forecast(P[-1], T, confidence);
    ES = model.expected_shortfall(T, confidence);
    return position_size(which_day, forecast, ES);
    
def main():
    cs.execute(F"select theday from prices where theday >= '2020-06-01';")
    days = [d[0] for d in cs.fetchall()]
    asset = {
        'old': np.nan,
        'new': np.nan
    };
    cash = {
        'old': np.nan,
        'new': np.nan
    };
    for d in days:
        asset['new'] = analyse(d)
        cs.execute(F"""
        select quantity, cash from positions
        where time_of_trade < '{d}'
        order by time_of_trade desc
        limit 1;
        """);
        asset['old'], cash['old'] = cs.fetchall()[0];
        cs.execute(F"""
        select price from prices
        where theday <= '{d}'
        order by theday desc
        limit 1;
        """);
        latest = cs.fetchall()[0][0]
        if round(asset['new']) != 0:
            cash['new'] = cash['old'] - asset['new'] * latest;
            cs.execute(F"""
            insert into positions values
            ('{d}', 'SP500', {asset['new']}, {cash['new']});
            """);
        conn.commit();
    
if __name__ == "__main__":
    with closing(sqlite3.connect("SP500.db")) as conn:
        with closing(conn.cursor()) as cs:
            prepare()
            main()
    
