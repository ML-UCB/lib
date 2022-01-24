#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:23:43 2022

@author: yajingleo
"""

import io
import unittest
import pandas as pd

import simulation

class mock_simulation(simulation.MultiPlayerMultiArmBanditSim):
    
    def populate_user(self) -> int:
        return 0
    
    def select_item(self, user: int) -> int:
        return 0
    
    def sim_rating(self, user: int, item: int) -> float:
        return 10
    

class TestMultiPlayerMultiArmBanditSim(unittest.TestCase):

    def setUp(self):
        config = simulation.MultiPlayerMultiArmBanditConfig(
            num_users = 10,
            num_items = 10,
            num_sims = 10,
            steps_to_log = 5
            )
        self.simulator = mock_simulation(config)

    def test_sim(self):
        self.simulator.sim()
        
        ratings_csv = """user,item,rating
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            0,     0,      10
            """
        expected_ratings_df = pd.read_csv(io.StringIO(ratings_csv))
        pd.testing.assert_frame_equal(self.simulator.ratings_df, 
                                      expected_ratings_df)
        
        regrets_csv = """step,regret
            5, 0.0
            """
        expected_regrets_df = pd.read_csv(io.StringIO(regrets_csv))
        pd.testing.assert_frame_equal(self.simulator.regrets_df, 
                                      expected_regrets_df)

if __name__ == '__main__':
    unittest.main()