import numpy as np
from pyomo.core import ConcreteModel
from scipy.stats import norm
from pyomo.environ import *
import numpy as np
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from openpyxl.workbook import Workbook
import matplotlib.pyplot as plt
# from _import_geographic_data import *
from matplotlib.offsetbox import AnchoredText
# from _import_optimization_files import *
import time
from pyomo.core.util import quicksum
from datetime import datetime
from pyomo.environ import *





"""Objective Function old"""
def minimize_waiting_and_charging(model: ConcreteModel):
    model.objective_function = Objective(
        expr=(
            quicksum(
                    model.n_wait[el] + model.n_wait_charge_next[el]
                    for el in model.charging_cells_key_set
            )
        ),
        sense=minimize,

    )