from abc import ABC, abstractmethod


class AbstractBPNAVEM(ABC):

    @abstractmethod
    def get_u(self, x):
        pass

    @abstractmethod
    def get_u_and_du(self, x):
        pass

    @abstractmethod
    def get_u0_phi_g(self, x):
        pass

    @abstractmethod
    def get_u0_phi_g_and_du0_dphi_dg(self, x):
        pass


