# PyCaMa: Python for multiobjective cash management with multiple bank accounts


from gurobipy import *
import numpy as np

class multibank(object):
    
    def __init__(self, banks, trans, A, gzero, gone, v, bmin):
        """Defines the cash management system"""
		
        self.banks = list(banks)               # List of banks
        self.trans = list(trans)               # List of transactions
        self.A = np.array(A, dtype= int)       # Incidence matrix  
        self.gzero = gzero                     # Dict of fixed transaction cost
        self.gone = gone                       # Dict of variable transaction costs
        self.v = v                             # Dict of holding costs  
        self.bmin = list(bmin)                 # List of minimum balances
        self.h = 1                             # Planning horizon (default to 1)
        self.resx = []                         # Optimal policy
        self.resb = []                         # Optimal balance
        self.objval = 0                        # Objective value
        self.costmax = 1                       # Maximum cost for multiobjective optimization
        self.riskmax = 1                       # Maximum cost for multiobjective optimization
        self.costref = 0                       # Cost reference for multiobjective optimization
        self.costweight = 1                    # Maximum cost for multiobjective optimization
        self.riskweight = 0                    # Cost reference for multiobjective optimization
        
		
        # Checks types of input data
        if type(self.gzero) != dict:
            self.gzero = dict([(i,0) for i in self.trans])
            self.A = np.zeros((len(self.trans), len(self.banks)))
            print("Fixed costs must be a dictionary")
        if type(self.gone) != dict:
            self.gone = dict([(i,0) for i in self.trans])
            self.A = np.zeros((len(self.trans), len(self.banks)))
            print("Variable costs must be a dictionary")
        if type(self.v) != dict:
            self.v = dict([(i,0) for i in self.banks])
            self.A = np.zeros((len(self.trans), len(self.banks)))
            print("Holding costs must be a dictionary")
        
        # Checks dimension agreement 
        if self.A.shape != (len(self.trans), len(self.banks)):
            self.A = np.zeros((len(self.trans), len(self.banks)))
            print("Incidence matrix dimensions do not agree with banks or transactions")
        if len(self.bmin) != len(self.banks):
            self.banks = []
            print("Minimum balances must agree with banks")
        if len(self.gzero) != len(self.gone):
            self.gzero = []
            self.gone = []
            print("Fixed and variable transaction costs must agree")
        if len(self.gzero) != len(self.trans):
            self.trans = []
            print("Transaction costs must agree with transactions")
        if len(self.v) != len(self.banks):
            self.banks = []
            print("Holding costs must agree with banks")
        if len(self.banks) <= 1:
            self.banks = []
            print("A system must have at least two banks")
			
    def describe(self):
        """Describe the main characteristics of the system"""
		
        print('Banks =', self.banks)
        print('Trans =', self.trans)
        print('Fixed costs =', self.gzero)
        print('Variable costs =', self.gone)
        print('Holding costs =', self.v)
        print('Minimum balances =', self.bmin)
        print('A =', self.A)
    
    def solvecost(self, b0, fcast):
        """Solve mba problem from initial balance b0 and forecast fcast
        fcast: an h x m matrix with h forecasts for m accounts
        b0: list with initial balances for each account"""
        
        # Reset solution values 
        self.resx = []                         # Optimal policy
        self.resb = []                         # Optimal cash balance
        self.objval = 0                        # Objective value
        
        # Checks dimensions
        fcast = np.array(fcast)
        if len(fcast) <= len(self.banks): 
            fcast = fcast.reshape((1,len(self.banks)))               # For one-step horizons
        self.h = fcast.shape[0]
        
        if len(b0) != len(self.banks):
            return (print("Dimension for minimum balances must agree with banks"))
        
        if fcast.shape != (self.h, len(self.banks)):
            return (print("Dimensions for forecasts must agree with horizon and banks"))
                    
        # Init model
        m = Model("example")

        #Ranges
        tr_range = range(len(self.trans))
        bk_range = range(len(self.banks))
        time_range = range(self.h)

        # Fixed costs: z = 1 if trans x occurs at time tau
        fixed = []
        for tau in time_range:
            fixed.append([])
            for t in self.trans:
                fixed[tau].append(m.addVar(obj = self.gzero[t], vtype = GRB.BINARY, name="z%d,%d" %(tau,t)))
        m.update()

        # Variable costs are proportional to transaction decision variables
        var = []
        for tau in time_range:
            var.append([])
            for t in self.trans:
                var[tau].append(m.addVar(obj = self.gone[t], vtype = GRB.CONTINUOUS, name="x%d,%d" %(tau,t)))
        m.update()

        # Holding costs are proportional to balance auxiliary decision variables
        bal = []
        for tau in time_range:
            bal.append([])
            for j in self.banks:
                bal[tau].append(m.addVar(obj = self.v[j], vtype = GRB.CONTINUOUS, name="b%d,%d" %(tau, j)))
        m.update()

        # Intitial transition constraints and minimum balance constraints
        for j in bk_range:
            m.addConstr(b0[j] + fcast[0][j] + LinExpr(self.A.T[j], var[0][:]) == bal[0][j], 'IniBal%d'% j)
            m.addConstr(bal[0][j] >= self.bmin[j], 'Bmin%s'%j)
        m.update()

        # Rest of transition constraints
        for tau in range(1, self.h):
            for j in bk_range:
                m.addConstr(bal[tau-1][j] + fcast[tau][j] + LinExpr(self.A.T[j],var[tau][:]) == bal[tau][j], 'Bal%d,%d'%(tau, j))
                m.addConstr(bal[tau][j] >= self.bmin[j], 'Bmin%d%d'%(tau,j))
        m.update()

        # Bounds and binary variables constraints
        K = 9999
	k = 0.001
        for tau in time_range:
            for i in tr_range:
                m.addConstr(var[tau][i] <= K*fixed[tau][i], name="cbig%d%d" %(tau, i))  # K is a very large number
		m.addConstr(var[tau][i] >= k*fixed[tau][i], name="csmall%d%d" %(tau, i))  # k is a very small number
        m.update() 

        # Optimization
        m.setParam('OutputFlag', 0) 
        m.modelSense = GRB.MINIMIZE
        m.optimize()

        # Checks if model is optimal and present results        
        if m.status == 2: 
            self.objval = m.ObjVal
            for dv in m.getVars():
                if 'x' in dv.varName:
                    self.resx.append([dv.varName, int(dv.x)])
                if 'b' in dv.varName:
                    self.resb.append([dv.varName, int(dv.x)])
            return(self.resx)
        else:
            return(print("I was unable to find a solution"))

			
    def solverisk(self, b0, fcast, c0, Cmax, Rmax, w1, w2):
        """Solve mba problem from initial balance b0 and forecast fcast
        fcast: an h x m matrix with h forecasts for m accounts
        b0: list with initial balances for each account
		"""
        
        # Reset solution values 
        self.resx = []                         # Optimal policy
        self.resb = []                         # Optimal cash balance
        self.objval = 0                        # Objective value
        self.costref = c0                      # Stores cost reference
        self.costmax = Cmax                    # Stores maximum cost
        self.riskmax = Rmax                    # Stores maximum risk
        self.costweight = w1                   # Weight for cost
        self.riskweight = w2                   # Weight for risk

        
        # Checks dimensions
        fcast = np.array(fcast)
        if len(fcast) <= len(self.banks): 
            fcast = fcast.reshape((1,len(self.banks)))               # For one-step horizons
        self.h = fcast.shape[0]
        
        if len(b0) != len(self.banks):
            return (print("Dimension for minimum balances must agree with banks"))
        
        if fcast.shape != (self.h, len(self.banks)):
            return (print("Dimensions for forecasts must agree with horizon and banks"))
                    
        # Init model
        m = Model("example")

        #Ranges
        tr_range = range(len(self.trans))
        bk_range = range(len(self.banks))
        time_range = range(self.h)

        # Fixed costs: z = 1 if trans x occurs at time tau
        fixed = []
        for tau in time_range:
            fixed.append([])
            for t in self.trans:
                fixed[tau].append(m.addVar(obj = self.gzero[t], vtype = GRB.BINARY, name = "z%d,%d" %(tau, t)))
        m.update()

        # Variable costs are proportional to transaction decision variables
        var = []
        for tau in time_range:
            var.append([])
            for t in self.trans:
                var[tau].append(m.addVar(obj = self.gone[t], vtype = GRB.CONTINUOUS, name = "x%d,%d" %(tau, t)))
        m.update()

        # Holding costs are proportional to balance auxiliary decision variables
        bal = []
        for tau in time_range:
            bal.append([])
            for j in self.banks:
                bal[tau].append(m.addVar(obj = self.v[j], vtype = GRB.CONTINUOUS, name="b%d,%d" %(tau, j)))
        m.update()
        
        # Deviational variables above a given cost
        devpos = []
        for tau in time_range:
            devpos.append(m.addVar(vtype=GRB.CONTINUOUS, name = "devpos%d" %tau))
        m.update()

       # Deviation constraints 
        tc = []
        hc = []
        for tau in time_range:
            tc.append(sum([self.gzero[t]*fixed[tau][t-1] + self.gone[t]*var[tau][t-1] for t in self.trans]))
            hc.append(sum([self.v[j]*bal[tau][j-1] for j in self.banks]))
            m.addConstr(tc[tau] + hc[tau] - devpos[tau] <= c0, 'DevCon%s' %tau)

        # Intitial transition constraints and minimum balance constraints
        for j in bk_range:
            m.addConstr(b0[j] + fcast[0][j] + LinExpr(self.A.T[j], var[0][:]) == bal[0][j], 'IniBal%d'% j)
            m.addConstr(bal[0][j] >= self.bmin[j], 'Bmin%s'%j)
        m.update()

        # Rest of transition constraints
        for tau in range(1, self.h):
            for j in bk_range:
                m.addConstr(bal[tau-1][j] + fcast[tau][j] + LinExpr(self.A.T[j],var[tau][:]) == bal[tau][j], 'Bal%d,%d'%(tau, j))
                m.addConstr(bal[tau][j] >= self.bmin[j], 'Bmin%d%d'%(tau,j))
        m.update()

        # Bounds and binary variables constraints
        K = 9999
	k = 0.001
        for tau in time_range:
            for i in tr_range:
                m.addConstr(var[tau][i] <= K*fixed[tau][i], name="cbig%d%d" %(tau, i))  # K is a very large number
		m.addConstr(var[tau][i] >= k*fixed[tau][i], name="csmall%d%d" %(tau, i))  # k is a very small number
        m.update() 

        # Setting the objectives
        transcost = sum([self.gzero[t] * fixed[tau][t-1] + self.gone[t] * var[tau][t-1] for tau in time_range for t in self.trans])
        holdcost = sum([self.v[j] * bal[tau][j-1] for tau in time_range for j in self.banks])
        devcost = sum([devpos[tau] for tau in time_range])
        m.setObjective((w1 / Cmax) * (transcost + holdcost)+(w2 / Rmax) * devcost, GRB.MINIMIZE)
        m.update()

        # Budget constraints
        m.addConstr(transcost + holdcost <= Cmax, name = "CostBudget")
        m.addConstr(devcost <= Rmax, name = "RiskBudget")
        m.update()

        # Optimization
        m.setParam('OutputFlag', 0) 
        m.optimize()

        # Checks if model is optimal and present results        
        if m.status == 2: 
            self.objval = m.ObjVal
            for dv in m.getVars():
                if 'x' in dv.varName:
                    self.resx.append([dv.varName, int(dv.x)])
                if 'b' in dv.varName:
                    self.resb.append([dv.varName, int(dv.x)])
            return(self.resx)
        else:
            return(print("I was unable to find a solution"))

			
    def policy(self): 
        """Returns a matrix with policy for each transaction"""
        if len(self.resx) > 0:
            plan = np.array(self.resx)
            planmat = np.array([int(i) for i in plan[:,1]]).reshape((self.h,len(self.trans)))
            return(planmat)
        else:
            return(print("Nothing to show"))
    
    def balance(self):
        """Returns a matrix with balances for each bank account"""
        if len(self.resb) > 0:
            bals = np.array(self.resb)
            balsmat = np.array([int(i) for i in bals[:,1]]).reshape((self.h,len(self.banks)))
            return(balsmat)
        else:
            return(print("Nothing to show"))
