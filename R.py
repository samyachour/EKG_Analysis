import rpy2.robjects as robjects
import numpy as np
import rpy2


def source(R_file):
    '''
        This function read in the R source file
        input:
            R_file: the name of the file 
    '''
    robjects.r.source(R_file)

def getFunction(function_name):
    '''
        grab the R function from the R source file
        input:
            function_name: the name of the fucntion on the R script
        output:
            The R function

        Example:
        #get the R function named 'train'
        train = getFunction('train')
    '''
    return robjects.globalenv[function_name]
    
def vecterize(value_list):
    '''
        Vecterize a python list into a R vector
        input:
            value_list: a python list 
        output:
            An R vector
    '''
    return robjects.Vector(value_list)

def matrix(r_vector, nrow, ncol):
    '''
        Generate an R matrix
        input:
            r_vector: an R vector that we want to convert to matrix
            nrow: the row that we want
        output:
            a R matrix
    '''
    return robjects.r.matrix(r_vector, nrow=nrow, ncol=ncol)

def null():
    '''
        output a R null value
    '''
    return rpy2.rinterface.NULL

def functionSource(function):
    '''
        print the source R function
    '''
    print (function.r_repr())




