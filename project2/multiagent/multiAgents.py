# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        GhostPos = [States.getPosition() for States in newGhostStates]
        DisToGhost = [abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1])for pos in GhostPos]
        TotalDis = sum(DisToGhost)
        food = newFood.asList()
        DisFood = [abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1]) for pos in food]
        near = DisFood and min(DisFood) or 0
        totalTime = sum(newScaredTimes)
        grade = 10 / (1+newFood.count()) + 0.2*TotalDis+ 5 / (0.1*near +1) + 1.5*successorGameState.getScore() +10*totalTime

        return grade

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState,self.depth,0,None,gameState.getNumAgents())[1] 
        
        
        
        
    def value(self,gameState,depth,agent,direction,NumAgents):
        if gameState.isWin():
            return [self.evaluationFunction(gameState),direction]
        elif gameState.isLose():
            return [self.evaluationFunction(gameState),direction]
        if agent == NumAgents : agent = 0
        if depth == 0 and agent == 0: return [self.evaluationFunction(gameState),direction]
        else:
            if agent == 0:
                return self.maxagent(gameState,depth,agent,direction,NumAgents)
            else :
                return self.minagent(gameState,depth,agent,direction,NumAgents)
        
    def maxagent(self,gameState,depth,agent,direction,NumAgents):
        v = [-99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.value(next,depth - 1,agent + 1,action,NumAgents)[0]
            if newvalue > v[0] : 
                v[1] = action
                v[0] = max(v[0],newvalue)
        return v        
        
    def minagent(self,gameState,depth,agent,direction,NumAgents):
        v = [+99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.value(next,depth ,agent + 1,action,NumAgents)[0]
            if newvalue < v[0] : 
                v[1] = action
                v[0] = min(v[0],newvalue)
        
        return v        








class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState,self.depth,0,None,gameState.getNumAgents(),-99999,99999)[1]
        
        
    def value(self,gameState,depth,agent,direction,NumAgents,alpha,beta):
        if gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState),direction]
        if agent == NumAgents : agent = 0
        if depth == 0 and agent == 0: return [self.evaluationFunction(gameState),direction]
        else:
            if agent == 0:
                return self.maxagent(gameState,depth,agent,direction,NumAgents,alpha,beta)
            else :
                return self.minagent(gameState,depth,agent,direction,NumAgents,alpha,beta)
        
    def maxagent(self,gameState,depth,agent,direction,NumAgents,alpha,beta):
        v = [-99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.value(next,depth - 1,agent + 1,action,NumAgents,alpha,beta)[0]
            if newvalue > v[0] : 
                v[1] = action
                v[0] = max(v[0],newvalue)
            if newvalue > beta:
                return v
            alpha = max(v[0],alpha)
            
        return v        
        
    def minagent(self,gameState,depth,agent,direction,NumAgents,alpha,beta):
        v = [+99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.value(next,depth ,agent + 1,action,NumAgents,alpha,beta)[0]
            if newvalue < v[0] : 
                v[1] = action
                v[0] = min(v[0],newvalue)
            if newvalue < alpha:
                return v
            beta = min(v[0],beta)
            
        
        return v        






class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState,self.depth,0,None,gameState.getNumAgents())[1] 
        
        
        
        
    def value(self,gameState,depth,agent,direction,NumAgents):
        if gameState.isWin():
            return [self.evaluationFunction(gameState),direction]
        elif gameState.isLose():
            return [self.evaluationFunction(gameState),direction]
        if agent == NumAgents : agent = 0
        if depth == 0 and agent == 0: return [self.evaluationFunction(gameState),direction]
        else:
            if agent == 0:
                return self.maxagent(gameState,depth,agent,direction,NumAgents)
            else :
                return self.expectagent(gameState,depth,agent,direction,NumAgents)
        
    def maxagent(self,gameState,depth,agent,direction,NumAgents):
        v = [-99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.value(next,depth - 1,agent + 1,action,NumAgents)[0]
            if newvalue > v[0] : 
                v[1] = action
                v[0] = max(v[0],newvalue)
        return v        
        
    def expectagent(self,gameState,depth,agent,direction,NumAgents):
        v = [+99999,None]
        actions = gameState.getLegalActions(agent)
        total = []
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.value(next,depth ,agent + 1,action,NumAgents)[0]
            total.append(float(newvalue))
        
        v[0] = sum(total) / len(actions)
        return v       

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    GhostPos = [States.getPosition() for States in newGhostStates]
    DisToGhost = [abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1])for pos in GhostPos]
    TotalDis = sum(DisToGhost)
    food = newFood.asList()
    DisFood = [abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1]) for pos in food]
    near = DisFood and min(DisFood) or 0
    totalTime = sum(newScaredTimes)
    grade = 10 / (1+newFood.count()) + 0.2*TotalDis+ 5 / (0.1*near +1) + 1.5*successorGameState.getScore() +2*totalTime

    return grade

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        self.depth = 3
        return self.minimaxvalue(gameState,self.depth,0,None,gameState.getNumAgents(),-99999,99999)[1]
        
        
    def minimaxvalue(self,gameState,depth,agent,direction,NumAgents,alpha,beta):
        if gameState.isWin() or gameState.isLose():
            return [self.contestFunction(gameState),direction]
        if agent == NumAgents : agent = 0
        if depth == 0 and agent == 0: return [self.contestFunction(gameState),direction]
        else:
            if agent == 0:
                return self.maxagent(gameState,depth,agent,direction,NumAgents,alpha,beta)
            else :
                return self.minagent(gameState,depth,agent,direction,NumAgents,alpha,beta)
        
    def maxagent(self,gameState,depth,agent,direction,NumAgents,alpha,beta):
        v = [-99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.minimaxvalue(next,depth - 1,agent + 1,action,NumAgents,alpha,beta)[0]
            if newvalue > v[0] : 
                v[1] = action
                v[0] = max(v[0],newvalue)
            if newvalue > beta:
                return v
            alpha = max(v[0],alpha)
            
        return v        
        
    def minagent(self,gameState,depth,agent,direction,NumAgents,alpha,beta):
        v = [+99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.minimaxvalue(next,depth ,agent + 1,action,NumAgents,alpha,beta)[0]
            if newvalue < v[0] : 
                v[1] = action
                v[0] = min(v[0],newvalue)
            if newvalue < alpha:
                return v
            beta = min(v[0],beta)
            
        
        return v 
        
    def expectValue(self,gameState,depth,agent,direction,NumAgents):
        if gameState.isWin() or gameState.isLose():
            return [self.contestFunction(gameState),direction]
        if agent == NumAgents : agent = 0
        if depth == 0 and agent == 0: return [self.contestFunction(gameState),direction]
        else:
            if agent == 0:
                return self.maxagent(gameState,depth,agent,direction,NumAgents)
            else :
                return self.expectagent(gameState,depth,agent,direction,NumAgents)
        
    def maxagent(self,gameState,depth,agent,direction,NumAgents):
        v = [-99999,None]
        actions = gameState.getLegalActions(agent)
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.expectValue(next,depth - 1,agent + 1,action,NumAgents)[0]
            if newvalue > v[0] : 
                v[1] = action
                v[0] = max(v[0],newvalue)
        return v        
        
    def expectagent(self,gameState,depth,agent,direction,NumAgents):
        v = [+99999,None]
        actions = gameState.getLegalActions(agent)
        total = []
        for action in actions:
            next = gameState.generateSuccessor(agent, action)
            newvalue = self.expectValue(next,depth ,agent + 1,action,NumAgents)[0]
            total.append(float(newvalue))
        
        v[0] = sum(total) / len(actions)
        return v          
        
        
        
        
    def contestFunction(self,currentGameState):
        """
          Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
          evaluation function (question 5).
    
          DESCRIPTION: <write something here so we know what you did>
        """
        "*** YOUR CODE HERE ***"
        successorGameState = currentGameState
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        GhostPos = [States.getPosition() for States in newGhostStates]
        DisToGhost = [abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1])for pos in GhostPos]
        TotalDis = sum(DisToGhost)
        food = newFood.asList()
        DisFood = [abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1]) for pos in food]
        near = DisFood and min(DisFood) or 0
        totalTime = sum(newScaredTimes)
        grade = 10 / (1+newFood.count()) + 0.2*TotalDis+ 5 / (0.1*near +1) + 1.5*successorGameState.getScore() +2*totalTime
    
        return grade
   
    
        
