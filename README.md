# PyCaMa
Python for multiobjective cash management

Despite the recent advances in cash management, there is a lack of supporting software to aid the transition from theory to practice. In order to fill this gap, we provide a cash management module in Python for practitioners interested in either building decision support systems for cash management or performing their own experiments. 

Cash managers usually deal with multiple banks to receive payments from customers and to send payments to suppliers. Operating such a cash management system implies a number of transactions between accounts, what is called a policy, to maintain the system in a state of equilibrium, meaning that there exists enough cash balance to face payments and avoid an overdraft. In addition, optimal policies allow to keep the sum of both transaction and holding costs at a minimum. However, cash managers may be interested not only in cost but also the risk of policies. Hence, risk analysis can also be incorporated as an additional goal to be minimized in cash management. As a result, deriving optimal policies in terms of both cost and risk within systems with multiple bank accounts is not an easy task. PyCaMa is able to provide such optimal policies.

PyCaMa is a Python-Gurobi tool aimed to automate multiobjective decision-making in cash management. PyCaMa contributes to support scientific discovery in cash management by: (i) empowering cash managers to perform experiments, e.g., on the utility of forecasts; (ii) eliciting the best precautionary minimum balances; (iii) allowing an easy extension to a multiobjective approach by considering additional objectives such as the risk of the policies. In addition, PyCaMa support daily decision-making in cash management by providing a tool to derive optimal policies within a real-world context where cash management systems with multiple bank accounts are the rule rather than the exception.
