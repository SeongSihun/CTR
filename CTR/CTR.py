################################################################################################################
################################################################################################################
##  CTR.py
##  made by Sihun Seong


import json
import traceback
import pandas as pd
import numpy as np
import itertools

################################################################################################################

pi   = np.pi
inf  = np.inf
#
exp  = lambda x: np.exp(x)
#
def asin(x):
	x = np.round(x, 10)
	return np.arcsin(x)
#
vec  = lambda *args: np.array(args)
unit = lambda *args: vec(*args)/norm(vec(*args))
span = lambda eval, evec : np.tensordot(eval, evec, axes=0)
#
def norm(v):
	if len(v.shape)>1:
		return np.linalg.norm(v, axis=1)
	else: return np.linalg.norm(v)
def Crit(v):
	n = np.where(v==0, 1, v)
	v = np.where(v==0, 0, 1/n)
	v[(np.where(v==0)[0]+1)%3] *= -1
	return v
def Orthogonal(v):
	z = np.count_nonzero(v)
	if z==1:
		M = vec([0, 1, 0], [0, 0, 1], [1, 0, 0])
		return M@v, M@M@v
	if z==2: o = Crit(v)
	if z==3: o = Crit(vec(*v[0:2], 0))
	return (o, np.cross(v, o))

################################################################################################################

class Xray:
	CuKa1  = 1.5406
	CuKa2  = 1.544390
	def __init__(self, wavelength=CuKa1, Nq = 1000):
		(h, c, e) = (6.62607015E-34, 299792458, 1.6021773349E-19)
		self.wavelength = wavelength # Å unit
		self.Energy = h * c / (wavelength * 1E-10) / e
		self.k = 2 * pi / self.wavelength
		self.G = np.linspace(0, 2*self.k, Nq+1)[1:]
		self.ttheta = 2 * asin(self.G / 2 / self.k)
		self.degree = np.rad2deg(self.ttheta)
	def DEGREE(self, degree):
		self.ttheta = np.deg2rad(degree)
		self.degree = degree
		self.G = 2 * self.k * np.sin(self.ttheta / 2)
	def Q(self, *ref):
		q  = span(self.G, unit(*ref))
		self.q, self.qx, self.qy, self.qz = q, q[:,0], q[:,1], q[:,2]
		return self.q
	# def pseudoQ(self, *ref):
	# 	ref = vec([1, 1, 0], [-1, 1, 0], [0, 0, 1]) @ vec(*ref) / 2
	# 	return self.Q(*ref)
	# PQ = pseudoQ # Alias
	def HKL(self, molecule): return molecule.Q2HKL(self.q)
	def AFF(self, atom): return atom.aff(self.q, self.Energy)
	def SF(self,  molecule): return molecule.SF(self.q, self.Energy)
	def F(self, sample): return sample.F(self.q, self.Energy)
	def I(self, sample): return sample.I(self.q, self.Energy)

class Xray2d(Xray):
	def __init__(self, wavelength=Xray.CuKa1, Nq = 1000):
		super().__init__(wavelength, Nq)
		self.Gx = self.G
		self.Gy = self.G
	def DEGREE(self, degreeX, degreeY):
		self.tthetaX = np.deg2rad(degreeX)
		self.tthetaY = np.deg2rad(degreeY)
		self.Gx = 2 * self.k * np.sin(self.tthetaX / 2)
		self.Gy = 2 * self.k * np.sin(self.tthetaY / 2)
	def Q(self, *ref):
		self.ux, self.uy = Orthogonal(unit(*ref))
		self.x, self.y = np.meshgrid(self.Gx,self.Gy)
		self.q = (span(self.x, self.ux)+span(self.y, self.uy)).reshape(self.x.size, 3)
		return self.q
	# def HKL(self, molecule): return self.Gx * (self.ux @ molecule.abc) / 2 / pi, self.Gy * (self.uy @ molecule.abc) / 2 / pi
	def F(self, sample): return sample.F(self.q, self.Energy).reshape(self.x.shape)
	def I(self, sample): return sample.I(self.q, self.Energy).reshape(self.x.shape)

# class PowderXray(Xray):
# 	def F(self, film):
# 		F = np.complex128(np.zeros(len(self.degree)))
# 		arr = np.array([x/np.linalg.norm(x) for x in itertools.islice(itertools.product(*[range(int(x)) for x in 2 * self.k * film.molecule.abc / 2 / pi]), 1, None)])
# 		for q in np.unique(arr, axis=0):
# 			self.Q(*q)
# 			F += super().F(film)
# 		return F
# 	def I(self, film): return np.abs(self.F(film))**2


################################################################################################################

class Atom():
	# File path
	# https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
	PATH = "./CTR/sf"
	AFF = pd.read_csv(f"{PATH}/AFF0.csv", delimiter='\s+').apply(lambda x: x.strip() if isinstance(x, str) else x)
	#
	def __init__(self, Z, def_name=None):
		self.Z = Z
		self.f = None
		#
		if def_name == None:
			(_, _, _, text) = traceback.extract_stack()[-2]
			self.def_name = text[:text.find('=')].strip()
	def __call__(self, *args):
		return (self, *args)
	#
	def aff(self, Q, E=Xray().Energy):
		if self.f == None:
			f = pd.read_csv(Atom.PATH+f"/{self.def_name.lower()}.nff", delimiter='\s+')	
			differences = abs(f.iloc[:, 0] - E)
			closest_row = f.loc[[differences.idxmin()]]
			self.f = vec(closest_row.f1 + 1j * closest_row.f2)[0]
			# self.COEF = np.float64(Atom.AFF[Atom.AFF['Element'] == self.def_name].iloc[0].values[1:])
			self.COEF = Atom.AFF.iloc[np.where(np.char.strip([*Atom.AFF['Element']])==self.def_name)[0]].iloc[0,1:]
		(a1, b1, a2, b2, a3, b3, a4, b4, c) = np.float64(self.COEF)
		f0 = self.f + sum(c + vec(*[a * exp(-1 * b * np.power(norm(Q) / (4 * pi), 2)) for a, b in zip((a1, a2, a3, a4), (b1, b2, b3, b4))]))
		return f0

class Empty(Atom):
	def __init__(self):
		self.Z = None
		self.f = None
	def aff(self, Q, E):
		return np.zeros(len(Q))
#
class Molecule():
	def __init__(self, lattice, structure):
		self.lattice = np.array(lattice)
		self.abc = np.array(lattice[0:3])
		self.angle = np.array(lattice[3:])
		#
		self.structure = np.array(structure)
		self.atoms = self.structure[:,0]
		self.RJ = self.structure[:,1:]
		#
		self.map = self.LatticeMap()
		self.volume = np.linalg.det(self.map)
		self.rmap   = self.ReciprocalSpace()
	def __call__(self, *args): return self, args
	##
	def LatticeMap(self): # [a b c]
		# a, b, c = self.abc
		alpha, beta, gamma = tuple(map(np.deg2rad, self.angle))
		cx = np.cos(beta)
		cy = (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
		cz = np.sqrt(1 - np.power(cx,2) - np.power(cy,2))
		M = np.array([
			[1,np.cos(gamma),cx],
			[0,np.sin(gamma),cy],
			[0,0,cz]
		])
		return np.around(self.abc * M, 14)
	##
	def ReciprocalSpace(self): # [b1 b2 b3]
		b = np.zeros([3,3])
		for k in range(3):
			i = (k+1)%3
			j = (k+2)%3
			ai = self.map[:,i]
			aj = self.map[:,j]
			b[:,k] = 2*pi*np.cross(ai, aj) / self.volume
		return b
	##
	def Q2HKL(self, Q): return (np.linalg.inv(self.rmap) @ Q.T).T
	def HKL2Q(self, HKL): return (self.rmap @ HKL.T).T
	##
	def SF(self, Q, E):
		aff = vec(*[atom.aff(Q, E) for atom in self.atoms])
		phase = Q @ (1j * self.map @ self.RJ.T).astype(dtype=complex)
		# phase = (1j * self.RJ @ self.map @ Q.T).astype(dtype=complex)
		return sum(aff * exp(phase.T))
	def __mul__(self, tup):
		tup = vec(*tup)
		abc = self.abc * tup
		structure = self.structure.copy()
		structure[:,1:] = structure[:,1:] * tup
		return Molecule([*abc, *self.angle], structure)
	def __truediv__(self, substrate):
		substrate, ref = substrate
		lamda = self.volume / substrate.volume
		strain = lamda * unit(*ref)
		abc = substrate.abc * np.where(strain == 0, 1, strain)
		return Molecule(
			lattice = [*abc, *substrate.angle],
			structure = self.structure
	)	
	#### 사라질 예정
	def pseudocubic(*abc):
		a, b, c = abc
		ac      = np.sqrt(a**2 + b**2) / 2
		return vec(ac, ac, c/2)
		
class vdW(Molecule):
	def __init__(self, lattice):
		super().__init__(lattice, [Empty()(0,0,0)])

class SC(Molecule):
	def __init__(self, abc, X):
		structure = [
			X(0, 0, 0)
		]
		super().__init__([*abc, 90,90,90], structure)
	#
class BCC(Molecule):
	def __init__(self, abc, AB):
		A, B = AB
		structure = [
			A(0, 0, 0),
			B(0.5, 0.5, 0.5)
		]
		super().__init__([*abc, 90,90,90], structure)
	#
class FCC(Molecule):
	def __init__(self, abc, X):
		structure = [
			X(0, 0, 0),
			*[X(*rj) for rj in (np.ones([3, 3]) - np.eye(3))/2]
		]
		super().__init__([*abc, 90,90,90], structure)
	#
class Perovskite(Molecule):
	def __init__(self, abc, ABO):
		A, B, O = ABO
		structure = [
			A(0,0,0),
			B(0.5,0.5,0.5), #BCC
			*[O(*rj) for rj in (np.ones([3, 3]) - np.eye(3))/2],  #FCC
		]
		super().__init__([*abc, 90,90,90], structure)
#
class Film():
	def __init__(self, molecule, N):
		# self.molecule, self.ref = molecule
		self.molecule = molecule
		self.N = vec(*N)
	def __call__(self, *args): return Sample(self, nref=args)
	def __truediv__(self, substrate):
		# Film/Sample
		if 'Sample' in str(substrate.__class__):
			return Sample(self, *substrate.film, nref=substrate.nref)
		else:   # Film/Film
			return Sample(self, substrate, nref=None)
	def __or__(self, xray): 
		I = xray.I(self)
		return I/np.max(I)
	#
	def SN(self, Q): # X = Q @ molecule.abc
		# IX = 1j * Q * self.molecule.abc
		IX = 1j *  Q @ self.molecule.map
		# Numerator
		Nz = np.where(np.isinf(self.N), 0, self.N)
		NUM = np.where(np.isinf(self.N), -1, 1-exp(Nz * IX))
		NUM = np.where(1==exp(IX), Nz, NUM)
		# Denominator
		DEN = np.where(1==exp(IX), 1, 1-exp(IX))
		return np.prod(NUM/DEN, axis=1)
	#
	def F(self, Q, E=Xray().Energy): return self.molecule.SF(Q, E) * self.SN(Q)
	def I(self, Q, E=Xray().Energy): return np.abs(self.F(Q, E)) ** 2

class RoughCut(Film):
	def __init__(self, molecule, N, beta, nref=(0,0,0)):
		super().__init__(molecule, N)
		self.nref = vec(*nref)
		self.beta = beta
		
	def SN(self, Q): # X = Q @ molecule.abc
		# IX = 1j * Q * self.molecule.abc
		IX = 1j *  Q @ self.molecule.map
		expIX = exp(IX)
		Nz = np.where(np.isinf(self.N), 0, self.N)
		NUM = np.where(expIX==1, Nz, 1-exp(IX*Nz))
		NUM = np.where(self.N == inf, -1, NUM)
		DEN = np.where(expIX==1, 1, 1-expIX)
		# Roughness
		NUMR = exp(Nz * self.nref) * np.where(self.beta * expIX==1, -1, self.beta * expIX)
		DENR = np.where(self.beta * expIX==1, 1, 1-self.beta*expIX)
		return np.prod(NUM/DEN + NUMR/DENR, axis=1)

class Sample():
	def __init__(self, *film, nref):
		self.film = film
		self.nref = nref
		# Roughness nref set
		if 'Rough' in str(self.film[0].__class__):
			self.film[0].nref = nref
	def __truediv__(self, substrate):
		# Sample/Sample
		if 'Sample' in str(substrate.__class__):
			return Sample(*self.film, *substrate.film, nref=substrate.nref)
		else:	# Sample/Film
			return Sample(*self.film, substrate, nref=None)
	# def __or__(self, xray): return self.I(xray.q, xray.Energy)
	def __or__(self, xray):
		I = xray.I(self)
		return I/np.max(I)

	#
	def F(self, Q, E):
		F = np.zeros(len(Q)).astype(dtype=complex)
		PHI = np.zeros_like(Q).astype(dtype=complex)
		for film in self.film:
			expNIX = np.prod(exp(PHI), axis=1)
			F += (expNIX * film.F(Q, E))
			Nz = np.where(np.isinf(film.N), 0, film.N)
			# Nz = np.where(film.N==-inf, 0, Nz)
			IX = 1j *  Q @ film.molecule.map
			PHI += IX * (Nz * self.nref)
		return vec(*F)
	#
	def I(self, Q, E=Xray().Energy): return np.abs(self.F(Q, E)) ** 2
	



__all__ = ['Xray', 'Xray2d', 'Atom', 'Molecule', 'vdW','SC', 'FCC', 'BCC', 'Perovskite', 'Film', 'RoughCut', 'Sample']
