import math

from config import *
from scripts.butil.seq_config import *
from scripts.butil.eval_results import *
from scripts.butil.load_results import *
from scripts.butil.shift_bbox import *
from scripts.butil.split_seq import *
from scripts.butil.calc_seq_err_robust import *
from scripts.butil.calc_rect_center import *

def d_to_f(x):
    return [round(float(o),4) for o in x]

def matlab_double_to_py_float(double):
    return list(map(d_to_f, double))

def ssd(x, y):
    if len(x) != len(y):
        sys.exit("cannot calculate ssd")
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i])**2
    return math.sqrt(s)
