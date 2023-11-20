import numpy as np
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import roots_jacobi

from deepxde import config

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

        return coord_quadrature, weight_quadrature
    
    def gauss_legendre(self):
        
        if self.dimension == 1:
            try:
                self.ngp >= 1
                return leggauss(self.ngp)
            except:
                raise ValueError("Number of gauss points must be >=1, chosen is {self.ngp}<1")
    
    def gauss_labotto(self):
        
        if self.dimension == 1:
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
            
            return coord_quadrature, weight_quadrature
    
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
                     "2" : modified_legendre_2}
    
    dict_legendre_derivative = {"1" : modified_legendre_derivative,
                                "2" : modified_legendre_derivative_2}
    
    test_function = []
    test_function_derivative = []
    
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
    
    return d1 # d2 what is d2?
        