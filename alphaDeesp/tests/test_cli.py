import unittest
import subprocess

import numpy as np
import pandas as pd
import pathlib

from alphaDeesp import main
#from chronix2grid import constants as cst
#import chronix2grid.generation.generation_utils as gu


class TestCli(unittest.TestCase):

    def test_cli(self):
        cmd = ['expertop4grid',
            '-l', str(9),
            '-s',str(0),
            '-c',str(0),
            '-t',str(0)]

        rv = subprocess.run(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        self.assertEqual(rv.returncode, 0)
