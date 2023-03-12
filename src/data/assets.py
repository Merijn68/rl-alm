# Bank Assets
import pandas as pd

class Mortgage():
    """ Customer mortgage """
    def __init__(self):
        self.principal = 0
        self.interest_rate = 0
        self.years = 30
        self.fixed_period = 10
        

    def generate_mortgage_cashflow(principal, interest_rate, years, fixed_period):
        n = years * 12  # number of mortgage payments
        periods = fixed_period * 12
        principal_payment = principal / n  # monthly principal payment
        r = interest_rate / 12  # monthly interest rate
        cashflow = []
        cashflow.append((0, -principal))
        for i in range(1, periods + 1):
            interest_paid = principal * r
            if i == periods:
                principal_payment = principal
            principal = principal - principal_payment
            cashflow.append((i, principal_payment + interest_paid))
        cashflow_df = pd.DataFrame(cashflow, columns=["period", "cashflow"])
        return cashflow_df
    

class Bankmodel():
    """ Cashflow model of a bank """
    def __init__(self):
        




    
        