from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def act(self, s):

        ...

    @abstractmethod
    def update(self, s, sprime, a, r, done):

        ...

    @abstractmethod
    def finish_episode(self, episode_num):

        ...