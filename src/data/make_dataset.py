"""
    Create data sets from raw files to work with

"""

from pathlib import Path
import pandas as pd
from pandas import bdate_range   # business days
import numpy_financial as npf
import numpy as np
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import DateOffset


from loguru import logger

from random import randrange
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

def random_date(
        start: datetime, 
        end: datetime
) -> datetime:
    """ return a random business date between start and end date """    
    delta = end-start
    date = start + datetime.timedelta(days=randrange(delta.days))    
    # shift dates to next business day if closed
    if date.weekday() >= 5:
        date = date + BDay(1)
        date = date.to_pydatetime()
    return date

def get_interest_data(
        path: Path = Path("../data/raw/bonds.csv"),
) -> pd.DataFrame:
    """ Get interest rates dataset from DNB Statistics """

    logger.info("getting interest data from file ")
    df = pd.read_excel("../data/raw/interest.xlsx", parse_dates = ['Periode '], index_col = 'Periode ')
    df.index.rename('period',inplace=True)    
    df = df.query("Instrument == 'Woninghypotheken - Totaal ' " )
    df = df.query("Stroomtype == 'Nieuwe contracten '")
    df = df.query("RenteVastPeriode != 'Totaal * '")
    df = df[['RenteVastPeriode','waarde']]    
    df.columns = ['fixed_period', 'interest']    # subset columns
    df['fixed_period'] = df['fixed_period'].replace(
        {'> 1 jaar en <= 5 jaar * '  :   '1>5 years',
        '> 5 jaar en <= 10 jaar * ' :  '5>10 years',
        'Variabel en <= 1 jaar * '  : '<= 1 year', 
        '> 10 jaar * '              :  '> 10 years'        
        }
    )       
    df['fixed_period'] = df['fixed_period'].astype("category")
    df['fixed_period']  = df['fixed_period'].cat.set_categories(['<= 1 year', '1>5 years', '5>10 years', '> 10 years' ], ordered=True)    
    return df


def date_schedule(
  start_date: str,
  n_periods : int      
) -> np.array:
    """ Generate monthly payment dates schedule """
    
    # create a date range with monthly frequency starting at start_date
    dates = pd.date_range(start=start_date, periods=n_periods, freq=DateOffset(months = 1))

    # shift dates to next business day if closed
    return dates.map(lambda x: x + BDay(1) if x.weekday() >= 5 else x)

def generate_mortgage_cashflow(principal, interest_rate, years, fixed_period
) -> pd.DataFrame:
    ''' Generate mortgage cashflows using repricing method for lineair interest model '''
    
    # Calculate the number of interest payments
    n = years * 12
    principal_payment = principal / n
    # Calculate the monthly interest rate
    r = interest_rate / 12            
    # Create an dic to store the cashflow
    cashflow = []
    periods = fixed_period*12    
    # Loop through each payment period
    for i in range(1, periods+1):
        # Calculate the interest paid for the current period
        interest_paid = principal * r                
        # Subtract the principal paid from the remaining principal
        principal = principal - principal_payment        
        # Add the interest paid and remaining principal at interest fixing
        if i == periods:
            principal_payment = principal_payment +  principal        
        cashflow.append((i, principal_payment + interest_paid))        
    
    # Convert the cashflow list to a pandas DataFrame
    cashflow_df = pd.DataFrame(cashflow, columns=["Period", "cashflow"])
    
    return cashflow_df

def generate_mortgage_cashflows( 
        contracts : pd.DataFrame
) -> pd.DataFrame:
    
    mortgage_tenor = 30    
    dfs = []
    for index, row in contracts.iterrows():
        cf =  generate_mortgage_cashflow(row.principal, row.interest, mortgage_tenor, row.years)      
        cf['value_dt'] =  date_schedule(row.start_date, len(cf) )
        cf['contract'] = index
        dfs.append(cf)
    df_all = pd.concat([dfs[i] for i in range(len(contracts))], axis=0)    
    return df_all

def generate_mortgage_contracts(
        pos_date: str,
        n: int,
        df_i: pd.DataFrame
):
    # Generate mortgage contracts
    pos_date = parse(pos_date)    
    # probability of 10 years contracts is 10x higher then 1 year contract
    # as they can start 10 years before. 
    # Devision in fixed interest period is dependend on the coupon rates
    category = np.random.choice(  
                a=[0, 1, 2, 3], 
                size=n, 
                p=[0.03,                    
                   0.14, 
                   0.23,                    
                   0.60]  
                ) 
    df = pd.DataFrame()
    df['category'] = category
    d = {0: '<= 1 year', 1:  '1>5 years', 2: '5>10 years', 3:  '> 10 years'}    
    df['fixed_period'] = df['category'].map(d).astype('category')
    df['fixed_period'].cat.set_categories(['<= 1 year', '1>5 years', '5>10 years', '> 10 years' ], ordered=True)
    df['years'] = df['category'].map({0: 1, 1: 5,2: 10, 3: 20})
    df['start_date'] = df.apply(lambda row: random_date( pos_date -
                                relativedelta(years=row.years), pos_date), axis = 1)
    df['principal'] = 100000  
    df['period'] = df['start_date'].to_numpy().astype('datetime64[M]')  # trick to get 1th of the month
    df = df.merge(df_i.reset_index(),how = 'left', on = ['period','fixed_period'] )
    df['interest'].fillna(df_i['interest'].tail(1), inplace=True)    
    return df

