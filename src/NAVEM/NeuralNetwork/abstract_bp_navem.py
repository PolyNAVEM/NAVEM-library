from abc import ABC, abstractmethod


class AbstractBPNAVEM(ABC):

    @abstractmethod
    def setup_model_global_input(self, xy_per_pol, vertices, jac_per_pol, setup_n_derivatives, geometry_utilities):
        pass

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
    def get_u0_phi_g_and_du0_d_phi_dg(self, x):
        pass

    @abstractmethod
    def load_model(self, name_storage: str):
        pass

    @abstractmethod
    def save_model(self):
        pass
