import numpy as np
import json
import traceback
import pandas as pd


pi   = np.pi
inf  = np.inf
#
exp  = lambda x: np.exp(x)
#
def norm(vec):
    if len(vec.shape)>1:
        return np.linalg.norm(vec, axis=1)
    else: return np.linalg.norm(vec)
#
vec  = lambda *args: np.array([*args])
unit = lambda *args: vec(*args)/norm(vec(*args))
span = lambda eval, evec : np.tensordot(eval, evec, axes=0)



class Xray:
    CuKa1  = 1.5406
    CuKa2  = 1.544390
    
    ttheta = np.linspace(0, pi, 1024)[1:]
    TTHETA = np.rad2deg(ttheta)

    def __init__(self, wavelength=CuKa1, Energy=None):
        (h, c, e) = (6.62607015E-34, 299792458, 1.6021773349E-19)
        if Energy==None:
            self.wavelength = wavelength # Å unit
            self.Energy = h * c / (wavelength * 1E-10) / e
        else:
            self.Energy = Energy
            self.wavelength = h * c / (Energy * 1E-10) / e
        self.k = 2 * pi / self.wavelength

    def Q(self, *ref):
        self.G = 2 * self.k * np.sin(Xray.ttheta/2)
        self.Q = span(self.G, unit(*ref))
        return self.Q

    def HKL(self, molecule):
        return self.Q * molecule.abc / 2 / pi

    def tthetaChange(TTHETA=np.rad2deg(np.linspace(0, pi, 1024)[1:])):
        Xray.ttheta = np.deg2rad(TTHETA)
        Xray.TTHETA = TTHETA


class Atom():
    # File path
    # https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    PATH = "./DATA" 
    with open(PATH+"/AFF_F0.json", "r") as f:
        AtomicFormFactor = json.load(f)
    #
    def __init__(self, Z):
        self.Z = Z
        self.__den__ = 1
        self.f = None
        #
        (_, _, _, text) = traceback.extract_stack()[-2]
        self.def_name = text[:text.find('=')].strip()
    def __truediv__(self, num):
        atom = Atom(self.Z)
        atom.def_name = self.def_name
        atom.__den__ = num
        return atom
    def __call__(self, *args):
        return (self, *args)
    #
    def aff(self, Q, E=Xray().Energy):
        if self.f == None:
            f = pd.read_csv(Atom.PATH+f"/sf/{self.def_name.lower()}.nff", delimiter='\s+')	
            differences = abs(f.iloc[:, 0] - E)
            closest_row = f.loc[[differences.idxmin()]]
            self.f = vec(closest_row.f1 + 1j * closest_row.f2)[0]

        (a1, b1, a2, b2, a3, b3, a4, b4, c) = self.AtomicFormFactor[self.def_name]
        f0 = self.f + sum(c + vec(*[a * exp(-1 * b * np.power(norm(Q) / (4 * pi), 2)) for a, b in zip((a1, a2, a3, a4), (b1, b2, b3, b4))]))
        return f0 / self.__den__

class Molecule():
    def __init__(self, abc, structure):
        self.abc = vec(*abc)
        self.structure = vec(*structure)
        self.atoms = self.structure[:,0]
        self.RJ = self.structure[:,1:] * self.abc
    def __truediv__(self, substrate):
        return Molecule(
            abc = substrate.const_volume_abc(self.abc),
            structure = self.structure
    )
    def __add__(self, molecule):
        return Molecule(max(self.abc, molecule.abc), [*self.structure, *molecule.structure])
    def __call__(self, *args): return self, args
    #
    def pseudocubic(*abc):
        a, b, c = abc
        ac      = np.sqrt(a**2 + b**2) / 2
        return vec(ac, ac, c/2)
    def const_volume_abc(self, film_abc):
        a, b, _ = self.abc
        return np.array([a, b, np.prod(film_abc) / (a*b)])
    #
    def Q2HKL(self, Q):
        return Q * self.abc / 2 / pi
    def SF(self, Q, E=Xray().Energy):
        aff = vec(*[atom.aff(Q, E) for atom in self.atoms])
        phase = (1j * self.RJ @ Q.T).astype(dtype=complex)
        return sum(aff * exp(phase))

class SC(Molecule):
    def __init__(self, abc, X):
        structure = [
            X(0, 0, 0)
        ]
        super().__init__(abc, structure)

class BCC(Molecule):
    def __init__(self, abc, AB):
        A, B = AB
        structure = [
            A(0, 0, 0),
            B(0.5, 0.5, 0.5)
        ]
        super().__init__(abc, structure)

class FCC(Molecule):
    def __init__(self, abc, X):
        structure = [
            X(0, 0, 0),
            *[X(*rj) for rj in (np.ones([3, 3]) - np.eye(3))/2]
        ]
        super().__init__(abc, structure)

class Perovskite(Molecule):
    def __init__(self, abc, ABO):
        A, B, O = ABO
        structure = [
            A(0.5, 0.5, 0.5), 	#BCC
            B(0,0,0),
            *[O(*rj) for rj in (np.ones([3, 3]) - np.eye(3))/2],  #FCC
            # *[(B/8)(n >> 2 & 1, n >> 1 & 1, n & 1) for n in range(8)],
            # *[(O/2)(*rj) for rj in (np.ones([6,3])+vec(*np.eye(3), *-np.eye(3)))/2 ]
        ]
        super().__init__(abc, structure)

class Film():
    def __init__(self, molecule, N):
        # self.molecule, self.ref = molecule
        self.molecule = molecule
        self.N = vec(*N)
    def SN(self, Q): # X = Q @ molecule.abc
        IX = 1j * Q * self.molecule.abc

        Noinf = np.where(self.N == inf, 0, self.N)

        NUM = np.where(exp(IX)==1, Noinf, 1-exp(IX*Noinf))
        NUM = np.where(self.N == inf, -1, NUM)

        DEN = np.where(exp(IX)==1, 1, 1-exp(IX))
        return np.prod(NUM/DEN, axis=1)
    def I(self, Q, E=Xray().Energy): return np.abs(self.molecule.SF(Q, E) * self.SN(Q)) ** 2


class Sample():
    def __init__(self, *films, nref=vec(0,0,1)):
        # self.FILMS = films
        self.substrate = films[-1]
        self.film = films[0:-1]
        self.nref = nref

    def I(self, Q, E=Xray().Energy):
        I = self.substrate.I(Q, E)
        PHI = np.zeros_like(Q).astype(dtype=complex)
        for film in self.film:
            IX  = 1j * Q * film.molecule.abc
            expNIX = np.abs(np.prod(exp(PHI), axis=1)) ** 2
            I += (expNIX * film.I(Q, E))
            PHI += IX * (film.N * self.nref)
        return vec(*I)


__all__ = ['inf', 'vec', 'Xray', 'Atom', 'Molecule', 'SC', 'FCC', 'BCC', 'Perovskite', 'Film', 'Sample']

