#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:46:27 2022

@author: yajingleo
"""

"""A package for Multi-Arm Bandit simulations for recommendation systems.

Classical multi-arm bandit problem has one player and multiple games with 
random outputs. Here, we have a different settings in recommendation systems.

user: A sequence of users coming to consume items.
item: A list of available items.
reward: Given a pair of (user, item), the rating is the reward. 

Problem: For each coming user select an item for the user to maximize the
expected reward. 

Regret: Expectation(Best Reward - Select Reward)
"""

import abc
import attr

import pandas as pd

from typing import Optional

@attr.s
class MultiPlayerMultiArmBanditConfig(object):
    """A universal data class for simulation."""
    num_users = attr.ib(type=int)
    num_items = attr.ib(type=int)
    num_sims = attr.ib(type=int)
    steps_to_log = attr.ib(type=int)
    

class MultiPlayerMultiArmBanditSim(metaclass=abc.ABCMeta):
    """An interface for  Multi-Arm Bandit simulations for multiple players."""
    
    def __init__(self, config: MultiPlayerMultiArmBanditConfig):
        self._config = config
        self._ratings_df = pd.DataFrame()
        self._regrets_df = pd.DataFrame()
        self._step = 0
      
    @abc.abstractmethod
    def populate_user(self) -> int:
        pass
   
    @abc.abstractmethod
    def select_item(self, user: int) -> int:
        pass
    
    @abc.abstractmethod
    def sim_rating(self, user: int, item: int) -> float:
        pass


    def _compute_regret(self) -> Optional[float]:
        if self._ratings_df.size == 0:
            return None
        
        avg_rating = self._ratings_df.groupby(["user"])["rating"].mean()
        best_rating = self._ratings_df.groupby(["user", "item"])["rating"]\
            .mean().groupby(["user"]).min()
            
        return (best_rating - avg_rating).mean()
    
    def _log_rating(self, user: int, item: int, rating: float):
        self._ratings_df = self._ratings_df.append(
            [{"user":user, "item":item, "rating":rating}], ignore_index=True)
        
    def _log_regret(self, regret: Optional[float]):
        if regret is None:
            return 
        
        self._regrets_df = self._regrets_df.append(
            [{"step": self._step, "regret": regret}], ignore_index=True)
    
    def sim(self):
        for i in range(self._config.num_sims):
            user = self.populate_user()
            item = self.select_item(user)
            rating = self.sim_rating(user, item)

            self._log_rating(user, item, rating)
            
            if i > 0 and i % self._config.steps_to_log == 0:
                self._log_regret(self._compute_regret())
            self._step += 1
            
    @property
    def ratings_df(self):
        return self._ratings_df
    
    @property
    def regrets_df(self):
        return self._regrets_df
        
    