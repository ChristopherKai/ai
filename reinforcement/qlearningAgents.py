# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        
        self.qTable = {}
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if not self.qTable.has_key(state) or not self.qTable[state].has_key(action):
            return 0.0
        return self.qTable[state][action]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        value = -9999
        if not self.qTable.has_key(state):
            return 0
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        for action in self.getLegalActions(state):
            value = max(value,self.getQValue( state, action))
        return value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        self.value = -9999
        self.act = None
        if not self.qTable.has_key(state):
            self.qTable[state] = {}
        if len(self.qTable[state]) == 0:
            return None
        for action in self.qTable[state]:
            if self.qTable[state][action] >= self.value:
                self.value = self.qTable[state][action]
                self.act = action
        return self.act

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not self.qTable.has_key(state):
            
            self.qTable[state] = {}
            for action in legalActions:
                self.qTable[state][action] = 0
        if len(legalActions) == 0:
            return None
        coin = util.flipCoin(self.epsilon)
        if coin == True :
            action = random.choice(legalActions)
        else:
            v = -9999
            for act in legalActions:
                if self.qTable[state][act] > v:
                    v = self.qTable[state][act]
                    action = act
                

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #print state,nextState
        if not self.qTable.has_key(state):
            self.qTable[state] = {}
            self.qTable[state][action] = 0
        elif not self.qTable[state].has_key(action):
            self.qTable[state][action] = 0
        #sprint self.qTable,"Need:",state,"action:",action
        sample = reward + self.discount * float(self.getValue(nextState))
        qvalue = self.qTable[state][action] + self.alpha * float((sample - self.qTable[state][action]))
        self.qTable[state][action] = qvalue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
        
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        total = 0
        for (sta , act),f in self.featExtractor.getFeatures(state,action).items():
            total += f * self.weights[(sta,act)]
        return total

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feats = self.featExtractor.getFeatures(state,action)
        diff = reward + self.discount * self.getValue(nextState) - self.getQValue(state,action)
        for (sta,act),weight in self.weights.items():
            self.weights[(sta,act)] = weight + self.alpha * diff * feats[(sta,act)]
            
        
        
        
        
        
        
    def getValue(self,state):
        legalActions = self.getLegalActions(state)
        v = -99999
        if len(legalActions) == 0:
            return 0
        for action in legalActions:
            #print self.getQValue(state,action)
            v = max(v,self.getQValue(state,action))
        #print v
        return v
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
