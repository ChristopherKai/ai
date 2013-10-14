# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.depth = 1
        self.qTable = {}
        self.vTable = {}
        for state in mdp.getStates():
            self.vTable[state] = 0
            self.qTable[state] = {}
            for action in mdp.getPossibleActions(state):
                
                self.qTable[state][action] = 0
        
        while self.depth < self.iterations + 1:
            self.tempTable = {}
            for state in mdp.getStates():
                self.stateValue = 0
                if not mdp.isTerminal(state):
                    self.stateValue = -9999
                    for action in mdp.getPossibleActions(state):
                        self.Qtotal = 0
                        for nextState,prob in mdp.getTransitionStatesAndProbs(state,action):
                            self.reward = mdp.getReward(state, action, nextState)
                            self.Qtotal += prob * (self.reward + self.discount * self.vTable[nextState])
                            #print "###state:",state,"Next",nextState,"reward:",self.reward,"Qtotal",self.Qtotal,"Value:",self.vTable[nextState]
                        self.qTable[state][action] = self.Qtotal
                        #print self.qTable[state][action]
                        self.stateValue = max(self.stateValue,self.qTable[state][action])
                else:
                    self.tempTable[state] = 0
                self.tempTable[state] = self.stateValue
            self.vTable = self.tempTable
            self.depth += 1
            
        for state in mdp.getStates():
            self.stateValue = -9999
            for action in mdp.getPossibleActions(state):
                self.Qtotal = 0
                for nextState,prob in mdp.getTransitionStatesAndProbs(state,action):
                    self.reward = mdp.getReward(state, action, nextState)
                    self.Qtotal += prob * (self.reward + self.discount * self.vTable[nextState])
                self.qTable[state][action] = self.Qtotal

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.vTable[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        return self.qTable[state][action]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action = None
        qvalue = -9999
        if self.mdp.isTerminal(state):
            return None
        for act in self.qTable[state]:
            if self.qTable[state][act] > qvalue:
                qvalue = self.qTable[state][act]
                action = act
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
