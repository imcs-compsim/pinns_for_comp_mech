import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi

from deepxde import config
from scipy.special import legendre

from numpy.polynomial.legendre import leggauss
from numpy import inf

class GaussQuadratureRule:
    def __init__(self, rule_name, dimension, ngp, **additional_params):
        '''
        Constructs the gauss quad rule based on dimension and number of gauss points. 
        Available rules:
            gauss_labotto
        
        Returns:
        coord_quadrature: coordinates of quadrature points
        weight_quadrature: corresponding weights  
        '''
        self.rule_name = rule_name
        self.dimension = dimension 
        self.ngp = ngp
        self.additional_params = additional_params
        
    def generate(self):
        
        rule_dic = {
            "gauss_labotto" : self.gauss_labotto,
            "gauss_legendre" : self.gauss_legendre
        }
        
        coord_quadrature, weight_quadrature = rule_dic[self.rule_name]()

        return coord_quadrature.reshape(-1,self.dimension), weight_quadrature.reshape(-1,self.dimension)
    
    def gauss_legendre(self):
        
        if self.dimension == 1:
            try:
                self.ngp >= 1
                return leggauss(self.ngp)
            except:
                raise ValueError("Number of gauss points must be >=1, chosen is {self.ngp}<1")
        elif self.dimension == 2:
            coord_quadrature_1d, weight_quadrature_1d = leggauss(self.ngp)
            
            coord_quadrature_x, coord_quadrature_y= np.meshgrid(coord_quadrature_1d,coord_quadrature_1d)
            coord_quadrature_2d=np.array((coord_quadrature_x.ravel(), coord_quadrature_y.ravel())).T
            
            weight_quadrature_x, weight_quadrature_y= np.meshgrid(weight_quadrature_1d,weight_quadrature_1d)
            weight_quadrature_2d=np.array((weight_quadrature_x.ravel(), weight_quadrature_y.ravel())).T
            
            #common_weight = weights[:,0]*weights[:,1]
            
            return coord_quadrature_2d, weight_quadrature_2d #common_weight      
    
    def gauss_labotto(self):
        
        if self.ngp < 3:
            raise ValueError("ngp has to be larger 3.") 
        
        a = self.additional_params.get("a",0)
        b = self.additional_params.get("b",0)
        
        coord_quadrature = roots_jacobi(self.ngp - 2, a + 1, b + 1)[0]

        Wl = (
            (b + 1)
            * 2 ** (a + b + 1)
            * gamma(a + self.ngp)
            * gamma(b + self.ngp)
            / (
                (self.ngp - 1)
                * gamma(self.ngp)
                * gamma(a + b + self.ngp + 1)
                * (self.jacobi_polynomial(self.ngp - 1, a, b, -1) ** 2)
            )
        )

        weight_quadrature = (
            2 ** (a + b + 1)
            * gamma(a + self.ngp)
            * gamma(b + self.ngp)
            / (
                (self.ngp - 1)
                * gamma(self.ngp)
                * gamma(a + b + self.ngp + 1)
                * (self.jacobi_polynomial(self.ngp - 1, a, b, coord_quadrature) ** 2)
            )
        )

        Wr = (
            (a + 1)
            * 2 ** (a + b + 1)
            * gamma(a + self.ngp)
            * gamma(b + self.ngp)
            / (
                (self.ngp - 1)
                * gamma(self.ngp)
                * gamma(a + b + self.ngp + 1)
                * (self.jacobi_polynomial(self.ngp - 1, a, b, 1) ** 2)
            )
        )

        weight_quadrature = np.append(weight_quadrature, Wr)
        weight_quadrature = np.append(Wl, weight_quadrature)
        coord_quadrature = np.append(-1, coord_quadrature)
        coord_quadrature = np.append(coord_quadrature, 1)
        
        if self.dimension == 1:
            return coord_quadrature, weight_quadrature
        
        elif self.dimension == 2:    
            coord_quadrature_x, coord_quadrature_y= np.meshgrid(coord_quadrature,coord_quadrature)
            coord_quadrature_2d=np.array((coord_quadrature_x.ravel(), coord_quadrature_y.ravel())).T
            
            weight_quadrature_x, weight_quadrature_y= np.meshgrid(weight_quadrature,weight_quadrature)
            weight_quadrature_2d=np.array((weight_quadrature_x.ravel(), weight_quadrature_y.ravel())).T
            
            return coord_quadrature_2d, weight_quadrature_2d
            
    
    @staticmethod
    def jacobi_polynomial(n, a, b, x):
        
        return jacobi(n, a, b)(x)

    @staticmethod
    def jacobi_polynomial_derivative(n, a, b, x, k):
        "return derivative of oder k"
        ctemp = gamma(a + b + n + 1 + k) / (2 ** k) / gamma(a + b + n+ 1)
        
        return ctemp * jacobi(n - k, a + k, b + k)(x)
    
def get_test_function_properties(n_test, coord_quadrature, approach="2"):
    """_summary_

    Args:
        n_test (_type_): _description_
        coord_quadrature (_type_): _description_
        approach (str, optional): _description_. Defaults to "approach_2".

    Returns:
        _type_: _description_
    """
    
    dict_legendre = {"1" : modified_legendre,
                     "2" : modified_legendre_2,
                     "3" : legendre_basis_shape_functions,
                     "4" : lagrange_basis_shape_functions}
    
    dict_legendre_derivative = {"1" : modified_legendre_derivative,
                                "2" : modified_legendre_derivative_2,
                                "3" : legendre_basis_shape_functions_derivative,
                                "4": lagrange_basis_shape_functions_derivative}
    
    test_function = []
    test_function_derivative = []
    
    if approach == "4":
        test_function = lagrange_basis_shape_functions(n_test, coord_quadrature)
        test_function_derivative = lagrange_basis_shape_functions_derivative(n_test, coord_quadrature)
    else:
        for i in range(1, n_test+1):
            test_function.append(dict_legendre[approach](i, coord_quadrature))
            test_function_derivative.append(dict_legendre_derivative[approach](i, coord_quadrature))
    
    return np.array(test_function).astype(config.real(np)), np.array(test_function_derivative).astype(config.real(np))

def modified_legendre(n, x):
    # based on VPINNs paper
    return GaussQuadratureRule.jacobi_polynomial(n + 1, 0, 0, x) - GaussQuadratureRule.jacobi_polynomial(n - 1, 0, 0, x)

def modified_legendre_2(n, x):
    # http://hyperphysics.phy-astr.gsu.edu/hbase/Math/legend.html
    
    def legendre_p(n,x):
        sum_p = 0
        M = n//2
        for m in range(0, M+1):
            numerator = np.math.factorial(2*n-2*m)
            denominator = 2**n * np.math.factorial(m) * np.math.factorial(n-m) * np.math.factorial(n-2*m)
            sum_p = sum_p + (-1)**m * numerator / denominator * x**(n-2*m)
        
        return sum_p
    
    return legendre_p(n+1,x) - legendre_p(n-1,x)

def modified_legendre_derivative_2(n, x):
    # http://hyperphysics.phy-astr.gsu.edu/hbase/Math/legend.html
    
    def legendre_p(n,x):
        sum_p = 0
        M = n//2
        contribution = 0
        for m in range(0, M+1):
            numerator = np.math.factorial(2*n-2*m)
            denominator = 2**n * np.math.factorial(m) * np.math.factorial(n-m) * np.math.factorial(n-2*m)
            if (n-2*m) != 0:
                contribution = (-1)**m * numerator / denominator * (n-2*m)*x**(n-2*m-1)
                sum_p = sum_p + contribution
        
        return sum_p
    
    return legendre_p(n+1,x) - legendre_p(n-1,x)

def modified_legendre_derivative(n, x):
    # https://dlmf.nist.gov/18.9
    if n == 1:
        d1 = ((1 + 2) / 2) * GaussQuadratureRule.jacobi_polynomial(1, 1, 1, x)
        d2 = ((1 + 2) * (1 + 3) / (2 * 2)) * GaussQuadratureRule.jacobi_polynomial(0, 2, 2, x)
    elif n == 2:
        d1 = ((2 + 2) / 2) * GaussQuadratureRule.jacobi_polynomial(2, 1, 1, x) - (
            (2) / 2
        ) * GaussQuadratureRule.jacobi_polynomial(2 - 2, 1, 1, x)
        d2 = ((2 + 2) * (2 + 3) / (2 * 2)) * GaussQuadratureRule.jacobi_polynomial(1, 2, 2, x)
    else:
        d1 = ((n + 2) / 2) * GaussQuadratureRule.jacobi_polynomial(n, 1, 1, x) - (
            (n) / 2
        ) * GaussQuadratureRule.jacobi_polynomial(n - 2, 1, 1, x)
        d2 = ((n + 2) * (n + 3) / (2 * 2)) * GaussQuadratureRule.jacobi_polynomial(n - 1, 2, 2, x) - (
            (n) * (n + 1) / (2 * 2)
        ) * GaussQuadratureRule.jacobi_polynomial(n - 3, 2, 2, x)
    
    return d1 # d2 what is d2? --> d2 is the second derivative

def legendre_basis_shape_functions(n,x):
    # https://www.juliafem.org/FEMBase.jl/v0.2/basis/
    if n==1:
        return 1/2*(1-x)
    if n==2:
        return 1/2*(1+x)
    if n>2:
        j=n-1
        P1 = GaussQuadratureRule.jacobi_polynomial(j, 0, 0, x)
        P2 = GaussQuadratureRule.jacobi_polynomial(j-2, 0, 0, x)
        return 1.0/np.sqrt(2.0*(2.0*j-1.0))*(P1-P2)

def legendre_basis_shape_functions_derivative(n,x):
    # https://www.juliafem.org/FEMBase.jl/v0.2/basis/
    if n==1:
        return -0.5*np.ones_like(x)
    if n==2:
        return 0.5*np.ones_like(x)
    if n>2:
        j=n-1
        P1 = GaussQuadratureRule.jacobi_polynomial_derivative(j,0,0,x,1)
        if (j-2)==0:
            P2=0
        else:
            P2 = GaussQuadratureRule.jacobi_polynomial_derivative(j-2,0,0,x,1)
        return 1.0/np.sqrt(2.0*(2.0*j-1.0))*(P1-P2)

def lagrange_basis_shape_functions(n, x):
    """
    Compute the Lagrange shape functions for nD linearly spaced elements.

    """
    if (n == 1) or (n == 2):
        node_coord = np.array([-1,1])
    else:
        node_coord = np.linspace(-1,1,n)
    shape_functions = []
    n_nodes = len(node_coord)

    for i in range(n):
        N_i = 1.0
        for j in range(n_nodes):
            if j != i:
                N_i *= (x - node_coord[j]) / (node_coord[i] - node_coord[j])
        
        # normalize it
        # N_i = N_i/N_i[np.abs(N_i).argmax()]
        shape_functions.append(N_i)

    return shape_functions

def lagrange_basis_shape_functions_derivative(n, x):
    """
    Compute the Lagrange shape functions derivatives for nD linearly spaced elements.

    """
    if (n == 1) or (n == 2):
        node_coord = np.array([-1,1])
    else:
        node_coord = np.linspace(-1,1,n)
    derivatives = []
    n_nodes = len(node_coord)

    for i in range(n):
        dN_i = 0.0

        if n <= 2:
            if i == 0:
                dN_i = -0.5*np.ones_like(x)
            elif i == 1:
                dN_i = 0.5*np.ones_like(x)
        else:
            for k in range(n_nodes):
                dN_i_inner = 1
                if k != i:
                    for j in range(n_nodes):
                        if j != i and j != k:
                            dN_i_inner *= (x - node_coord[j]) / (node_coord[i] - node_coord[j])*(1/(node_coord[i]-node_coord[k]))
                    dN_i = dN_i + dN_i_inner
        # normalize it
        # N_i = dN_i/dN_i[np.abs(dN_i).argmax()]
        
        derivatives.append(dN_i)

    return derivatives
        