from typing import Self
from donotation import do
import equinox as eqx

from recurrent.datarecords import InputOutput, OhoInputOutput
from recurrent.myrecords import GodState
from recurrent.mytypes import *
from recurrent.parameters import *
from recurrent.monad import *

from recurrent.util import prng_split
